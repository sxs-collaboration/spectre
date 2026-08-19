# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from spectre.Informer import unit_test_build_path
from spectre.support.BinDirectory import PYTHON_DIR_NAME
from spectre.support.Logging import configure_logging


class TestBinDirectory(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(unit_test_build_path(), "BinDirectory").resolve()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True)
        self.build_bin_dir = Path(unit_test_build_path()).parent.parent / "bin"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def resolved_sys_path(self, directory=None):
        """The 'sys.path' entries that the CLI wrapper puts Python on

        Runs the 'python-spectre' wrapper, which passes its arguments straight
        to the interpreter. With a 'directory', copies the wrapper there first
        and creates a package directory next to it, to check what the same
        script does elsewhere.
        """
        wrapper = self.build_bin_dir / "python-spectre"
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / PYTHON_DIR_NAME).mkdir(exist_ok=True)
            wrapper = directory / "python-spectre"
            shutil.copy(self.build_bin_dir / "python-spectre", wrapper)
        printed = subprocess.run(
            [str(wrapper), "-c", "import sys; print('\\n'.join(sys.path))"],
            capture_output=True,
            text=True,
            check=True,
        )
        # 'sys.path[0]' is the working directory for '-c'
        return printed.stdout.splitlines()[1:]

    def test_cli_finds_the_package_next_to_itself(self):
        # The wrapper reaches the Python package through its own location, so
        # a copy of it in a simulation's bin directory uses the package next to
        # the copy. The entries configured into it at build time follow, so the
        # package next to the wrapper wins.
        build_entries = self.resolved_sys_path()
        self.assertEqual(
            build_entries[0], str(self.build_bin_dir / PYTHON_DIR_NAME)
        )

        copied_dir = self.test_dir / "CopiedBin"
        copied_entries = self.resolved_sys_path(copied_dir)
        self.assertEqual(copied_entries[0], str(copied_dir / PYTHON_DIR_NAME))
        # Everything after the first entry is baked in at build time. In the
        # build directory the baked package entry duplicates the first and
        # Python drops it; in a copy it stays, followed by the same tail
        self.assertEqual(
            copied_entries[1:],
            [str(self.build_bin_dir / PYTHON_DIR_NAME)] + build_entries[1:],
        )

        # The canonical 'PYTHONPATH' that CMake hands to tests and to
        # 'LoadPython.sh' starts with the same package directory
        canonical = re.search(
            r"^export PYTHONPATH=(.*)$",
            (self.build_bin_dir / "LoadPython.sh").read_text(),
            re.MULTILINE,
        ).group(1)
        self.assertEqual(
            canonical.split(":")[0],
            str(self.build_bin_dir / PYTHON_DIR_NAME),
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
