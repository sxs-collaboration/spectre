# Distributed under the MIT License.
# See LICENSE.txt for details.

"""The bin directory that a segment runs from

We freeze a copy of the bin directory into the segment so that the segment can
run even if the build directory is deleted or changed. It contains everything
the job needs to run and to submit the next segment:

```
0000_Inspiral/
    bin/
        EvolveGhBinaryBlackHole
        python/
        spectre
        SubmitTemplate.sh
        ...
    Inspiral.yaml
    Submit.sh
0001_Inspiral/
    bin -> ../0000_Inspiral/bin
```

The `<build>/bin` directory has this same layout, so it is valid to symlink a
build directory instead of copying it for a segment.
"""

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

BIN_DIR_NAME = "bin"

# Freeze these files, plus the requested executables
FROZEN_FILES = (
    "spectre",
    "python-spectre",
    "python",
    "Machine.yaml",
    "SubmitTemplate.sh",
    "SubmitTemplateBase.sh",
)


def _staging(path: Path) -> Path:
    """Where a copy is assembled before it is moved to the 'path'"""
    return path.with_name(path.name + ".part")


def _remove(path: Path) -> None:
    """Delete the 'path', be it a symlink, a directory, or absent"""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _cmake_cache(bin_dir: Path) -> Optional[Path]:
    """The CMake cache of the build directory the bin_dir is in, or None"""
    cache_file = bin_dir.parent / "CMakeCache.txt"
    return cache_file if cache_file.is_file() else None


def _check_build_is_relocatable(bin_dir: Path) -> None:
    """Raise if the 'bin_dir' holds shared SpECTRE libraries

    Copying such executables would give a false sense of safety: they load the
    libraries out of the build directory whatever happens to the copy.
    """
    cache_file = _cmake_cache(bin_dir)
    if cache_file is None:
        return
    cache_entry = re.search(
        r"^BUILD_SHARED_LIBS:[^=]*=(.*)$",
        cache_file.read_text(),
        re.MULTILINE,
    )
    if not cache_entry:
        return
    if cache_entry.group(1).strip().upper() in ("ON", "TRUE", "YES", "1"):
        raise RuntimeError(
            f"The build directory '{cache_file.parent}' is configured with"
            " 'BUILD_SHARED_LIBS=ON'. Its executables load the SpECTRE"
            " libraries out of it, so a copy of them stops working as soon as"
            " the build directory changes, which defeats the purpose of the"
            " bin directory. Either reconfigure with 'BUILD_SHARED_LIBS=OFF'"
            " and rebuild, or schedule with '--no-freeze-bin', which"
            " links to the build directory instead of copying it. That works"
            " with shared libraries but ties the run to the build directory."
        )


@dataclass(frozen=True)
class BinDirectory:
    """A handle on a segment's bin directory

    The 'path' need not exist yet. The bin directory is created either by
    'freeze', which copies one, or by 'link', which symlinks to one. Executables
    enter through 'add'.
    """

    path: Path

    @classmethod
    def this(cls) -> "BinDirectory":
        """The bin directory that this script runs from"""
        # This file sits in bin/python/spectre/support/ (three levels deep).
        # Don't resolve symlinks, so this works also with PY_DEV_MODE.
        return cls(path=Path(__file__).parents[3])

    def freeze(self, source: Optional["BinDirectory"] = None) -> "BinDirectory":
        """Copy the 'FROZEN_FILES' of the 'source' here, replacing what is here

        Executables enter through 'add'. Symlinks are dereferenced. Either
        completes or leaves nothing behind.

        Arguments:
          source: Optional. The bin directory to copy. (Default: 'this()')

        Returns: This bin directory.
        """
        if not source:
            source = BinDirectory.this()
        source_dir = source.path.resolve()
        logger.info(f"Freeze bin directory '{self.path}' from '{source_dir}'")

        # Check before copying anything, which is expensive
        _check_build_is_relocatable(source_dir)

        # Assemble the copy in a staging directory, then move it into place
        staging = _staging(self.path)
        shutil.rmtree(staging, ignore_errors=True)
        try:
            staging.mkdir()
            for name in FROZEN_FILES:
                entry = source_dir / name
                if entry.is_dir():
                    shutil.copytree(
                        entry,
                        staging / name,
                        ignore=shutil.ignore_patterns("__pycache__"),
                    )
                elif entry.is_file():
                    shutil.copy2(entry, staging / name)
            _remove(self.path)
            os.replace(staging, self.path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self

    def link(self, target: "BinDirectory") -> "BinDirectory":
        """Symlink to the 'target', replacing what is here

        The link is relative and never points at another symlink.

        Returns: This bin directory.
        """
        target_dir = target.path.resolve()
        logger.info(f"Link bin directory '{self.path}' -> '{target_dir}'")
        _remove(self.path)
        # Remove any staging directory, which is a leftover from a failed freeze
        shutil.rmtree(_staging(self.path), ignore_errors=True)
        self.path.symlink_to(
            os.path.relpath(target_dir, self.path.parent.resolve()),
            target_is_directory=True,
        )
        return self

    def add(self, executable: Union[str, Path]) -> Path:
        """Copy the 'executable' unless it is already here

        Append-only, so a queued job keeps running the binary it was scheduled
        with. Raises if this resolves into a build directory, which holds what
        it built and nothing else.

        Returns: The path of the executable in this directory.
        """
        executable = Path(executable)
        target_dir = self.path.resolve()
        destination = target_dir / executable.name
        if destination.is_file():
            logger.debug(f"Executable is already frozen: '{destination}'")
            return destination
        if _cmake_cache(target_dir) is not None:
            raise RuntimeError(
                f"The executable '{executable.name}' is not in the build"
                f" directory '{target_dir.parent}' that this simulation links"
                " to. Build it there, or freeze the simulation's own bin"
                " directory with '--freeze-bin', which can hold executables"
                " from anywhere."
            )
        logger.info(f"Add executable to bin directory: '{destination}'")
        staging = _staging(destination)
        try:
            shutil.copy2(executable, staging)
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)
        return destination

    def executable(self, name: Union[str, Path]) -> Path:
        """The path of the executable 'name' in this directory"""
        return self.path / Path(name).name
