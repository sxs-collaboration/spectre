# Distributed under the MIT License.
# See LICENSE.txt for details.

"""The bin directory that a simulation runs from

A scheduled job runs unsupervised long after it was submitted, so it must not
depend on the build directory it was scheduled from: that gets recompiled,
switched to another branch, and eventually deleted. Everything the job needs
after submission is therefore copied into a 'bin' directory of the simulation,
with its support files next to it, and the job runs from there:

```
MySimulation/
    bin/
        MyExecutable
        spectre
        python/spectre/
    support/
        Machine.yaml
        SubmitTemplateBase.sh
        SubmitTemplate.sh
    Segment_0000/...
```

A build directory has this layout too, so the CLI finds the Python package and
the support files at the same place relative to itself in either. That is what
lets a copy of it work: nothing has to be translated.

The bin directory is created once per simulation and is never updated
implicitly: a file that is already there is kept rather than replaced.

This module knows nothing about scheduling. Deciding whether a run creates a bin
directory, which executables it needs, and what is recorded about it is the
caller's; 'Schedule' does that.
"""

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from spectre.support.DirectoryStructure import PipelineStep, Segment

logger = logging.getLogger(__name__)

# Layout of the directory that holds the CLI wrapper and the Python package.
# 'cmake/SpectreSetupPythonPackage.cmake' configures the wrapper with the same
# names.
BIN_DIR_NAME = "bin"
PYTHON_DIR_NAME = "python"
SUPPORT_DIR_NAME = "support"

# The support files, by name. 'support/CMakeLists.txt' configures these into
# '<build_dir>/support', and we copy them here so a scheduled job can find them
# independent of the build directory.
SUPPORT_FILES = ("Machine.yaml", "SubmitTemplateBase.sh", "SubmitTemplate.sh")
SUBMIT_SCRIPT_TEMPLATE = "SubmitTemplate.sh"


def _copy_to_dir(src_file: Path, dest_dir: Path) -> Path:
    """Copy the 'src_file' to the 'dest_dir', keeping the file name the same

    Returns the path to the new file.
    """
    if not src_file.is_file():
        raise FileNotFoundError(
            f"No file to copy at '{src_file}'. If this is a file of a "
            "simulation that was moved or renamed, the paths recorded in its "
            "scheduler context file are stale."
        )
    if not dest_dir.is_dir():
        raise NotADirectoryError(f"No directory to copy into at '{dest_dir}'.")
    if src_file.resolve().parent == dest_dir.resolve():
        return src_file
    dest = (dest_dir / src_file.name).resolve()
    if dest.exists():
        # Callers are expected to have decided what to do about this already,
        # so overwriting here would be a surprise
        raise OSError(f"File already exists at '{dest}'.")
    logger.debug(f"Copy file: {src_file} -> {dest}")
    shutil.copy(src_file, dest)
    return dest


def _check_build_is_relocatable(build_dir: Path) -> None:
    """Raise if the 'build_dir' was configured with shared SpECTRE libraries

    Copying such a build's executables would give a false sense of safety: they
    load the libraries out of the build directory whatever happens to the copy.

    Asks the build's 'CMakeCache.txt' rather than looking for library files,
    whose names differ between systems ('.so', '.dylib') while the cache entry
    does not. A directory with no cache was never configured, so there is
    nothing to check.
    """
    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.is_file():
        return
    cache_entry = re.search(
        r"^BUILD_SHARED_LIBS:[^=]*=(.*)$",
        cache_file.read_text(),
        re.MULTILINE,
    )
    if not cache_entry:
        return
    # The values CMake reads as true
    if cache_entry.group(1).strip().upper() in ("ON", "TRUE", "YES", "1"):
        raise RuntimeError(
            f"The build directory '{build_dir}' is configured with"
            " 'BUILD_SHARED_LIBS=ON'. Its executables load the SpECTRE"
            " libraries out of it, so they stop working when the build"
            " directory changes, which defeats the purpose of the bin"
            " directory. Reconfigure with 'BUILD_SHARED_LIBS=OFF' (the"
            " default), or schedule with '--no-create-bin' to run from the"
            " build directory at your own risk."
        )


@dataclass(frozen=True)
class BinDirectory:
    """A handle on a bin directory and the layout around it

    Derives the layout from the 'path', so a caller passes one handle around
    instead of parallel paths. Use 'this' for the one this code lives in, 'find'
    for the one a run directory belongs to, and 'create' to make a new one. The
    handle says nothing about whether the directory exists.
    """

    path: Path

    @property
    def spectre_cli(self) -> Path:
        """The SpECTRE CLI in this directory

        Its presence is what makes a directory a bin directory. It finds the
        Python package next to it (see 'cmake/SpectrePythonExecutable.sh').
        """
        return self.path / "spectre"

    @property
    def python_dir(self) -> Path:
        """Directory that holds the Python package"""
        return self.path / PYTHON_DIR_NAME

    @property
    def support_dir(self) -> Path:
        """Directory that holds the 'SUPPORT_FILES'

        Contains the files that a scheduled job needs after submission, such as
        the machine description and the submit script template (for
        resubmissions).
        """
        return self.path.parent / SUPPORT_DIR_NAME

    @classmethod
    def this(cls) -> "BinDirectory":
        """The bin directory that this code lives in

        '<build_dir>/bin' when running the CLI from a build directory, or the
        simulation's frozen bin directory when running the CLI from there.
        """
        # This file is at '<bin_dir>/python/spectre/support/BinDirectory.py',
        # so the bin directory is three levels up.
        # Symlinks are not resolved on the way up: with 'PY_DEV_MODE' the Python
        # files are symlinks into the source tree, but the directories holding
        # them are real.
        return cls(path=Path(__file__).parents[3])

    @classmethod
    def find(cls, start_dir: Union[str, Path]) -> Optional["BinDirectory"]:
        """The bin directory closest to the 'start_dir', if there is one

        Looks at the 'start_dir' and then at the enclosing directories, and
        returns the first hit, so the nearest bin directory wins. Callers should
        therefore start at the most specific directory of the run (the
        'run_dir').

        The search stays inside the simulation: it ascends out of a directory
        only if it's known to `DirectoryStructure.py` (i.e., a 'Segment' or a
        'PipelineStep'). The first directory that is neither is the simulation's
        root; it is checked, and the search stops there. So from
        'MySim/001_Inspiral/Segment_0003' it checks the segment, the step and
        'MySim', and no higher.

        A build directory's 'bin' is a hit like any other, so a run sitting
        directly in a build directory is meant to run from it.
        """
        directory = Path(start_dir).resolve()
        while True:
            bin_dir = cls(path=directory / BIN_DIR_NAME)
            # Accept the first bin dir that contains the CLI
            if bin_dir.spectre_cli.is_file():
                return bin_dir
            # Only keep ascending while we are inside the simulation's
            # structure
            if not (Segment.match(directory) or PipelineStep.match(directory)):
                return None
            if directory.parent == directory:
                return None
            directory = directory.parent

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        source: Optional["BinDirectory"] = None,
        add_executables: Sequence[Path] = (),
    ) -> "BinDirectory":
        """Create a bin directory at the 'path' from the 'source' installation

        Copies the 'spectre_cli' and the whole 'python_dir', puts the
        'SUPPORT_FILES' in the 'support_dir' next to it, and adds the
        'add_executables'. That is everything a scheduled job needs after
        submission.

        Creating a bin directory fails if the build is configured with
        'BUILD_SHARED_LIBS=ON', because such executables load shared libraries
        out of the build directory and would stop working when it changes.

        Arguments:
          path: Directory to create. Must not exist yet.
          source: Optional. The installation to copy from. Defaults to 'this'
            one, i.e., the build directory that is scheduling the simulation.
          add_executables: Executables to copy. Every executable that the
            simulation may run should be listed, including the ones of later
            pipeline steps, because a later step resolves its executable from
            this directory. Executables can also be added with 'add' after
            creation.

        Returns: The new bin directory.
        """
        bin_dir = cls(path=Path(path))
        if source is None:
            source = cls.this()
        build_dir = source.path.parent
        logger.info(
            f"Create bin directory '{bin_dir.path}' from build directory"
            f" '{build_dir}'"
        )

        # Check before copying anything, so a build that can't be copied fails
        # before it leaves a half-populated directory behind
        _check_build_is_relocatable(build_dir)

        bin_dir.path.mkdir(parents=True)

        _copy_to_dir(source.spectre_cli, bin_dir.path)

        # Dereference symlinks: with 'PY_DEV_MODE' the Python files are
        # symlinks into the source tree.
        shutil.copytree(
            source.python_dir,
            bin_dir.python_dir,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        # A build with no machine selected has no support files
        support_files = [
            source.support_dir / file_name
            for file_name in SUPPORT_FILES
            if (source.support_dir / file_name).is_file()
        ]
        if support_files:
            bin_dir.support_dir.mkdir(parents=True, exist_ok=True)
            for support_file in support_files:
                _copy_to_dir(support_file, bin_dir.support_dir)

        for executable in add_executables:
            bin_dir.add(executable)

        return bin_dir

    def add(self, src_file: Union[str, Path]) -> Path:
        """Copy the 'src_file' into this directory unless it is already there

        A file of the same name that is already here is kept rather than
        replaced, so a later segment or pipeline step runs what the first one
        put here.

        Returns the path to the file in this directory.
        """
        src_file = Path(src_file)
        if src_file.resolve().parent == self.path.resolve():
            return src_file
        already_there = self.path / src_file.name
        if already_there.exists():
            logger.debug(
                "Keep the file already in the bin directory:"
                f" '{already_there}' (instead of '{src_file}')"
            )
            return already_there
        return _copy_to_dir(src_file, self.path)

    def executable(self, executable: Union[str, Path]) -> Path:
        """This directory's copy of the 'executable', if it has one

        Unlike 'add' this only reads: it never puts the executable into the bin
        directory. Returns the 'executable' unchanged if this directory holds
        no copy of it.
        """
        in_bin_dir = self.path / Path(executable).name
        return in_bin_dir if in_bin_dir.is_file() else Path(executable)
