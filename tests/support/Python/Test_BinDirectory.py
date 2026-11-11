# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import stat
import unittest
from pathlib import Path

from spectre.Informer import unit_test_build_path
from spectre.support.BinDirectory import BIN_DIR_NAME, BinDirectory
from spectre.support.Logging import configure_logging


def _write_executable(path: Path, contents: str = "#!/bin/bash\n"):
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class TestBinDirectory(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(unit_test_build_path(), "BinDirectory").resolve()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True)

        # A stand-in for a build directory, laid out the way CMake composes one
        self.build_dir = self.test_dir / "Build"
        self.source = BinDirectory(self.build_dir / BIN_DIR_NAME)
        (self.source.path / "python/spectre/__pycache__").mkdir(parents=True)
        (self.source.path / "python/spectre/Schedule.py").write_text("# code\n")
        (self.source.path / "Machine.yaml").write_text("Machine:\n")
        (self.source.path / "SubmitTemplate.sh").write_text("template\n")
        for name in ["spectre", "python-spectre", "Inspiral", "Ringdown"]:
            _write_executable(self.source.path / name)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def segment(self, name: str) -> BinDirectory:
        """A bin directory handle in a fresh segment of 'Sim'"""
        segment_dir = self.test_dir / "Sim" / name
        segment_dir.mkdir(parents=True, exist_ok=True)
        return BinDirectory(segment_dir / BIN_DIR_NAME)

    def test_this(self):
        self.assertEqual(
            BinDirectory.this().path,
            Path(unit_test_build_path(), "../..", BIN_DIR_NAME).resolve(),
        )

    def test_freeze(self):
        bin_dir = self.segment("0000_Inspiral").freeze(self.source)
        self.assertEqual(
            sorted(entry.name for entry in bin_dir.path.iterdir()),
            [
                "Machine.yaml",
                "SubmitTemplate.sh",
                "python",
                "python-spectre",
                "spectre",
            ],
        )
        self.assertTrue((bin_dir.path / "python/spectre/Schedule.py").is_file())
        self.assertFalse((bin_dir.path / "python/spectre/__pycache__").exists())
        for parent, dirs, files in os.walk(bin_dir.path):
            for name in dirs + files:
                self.assertFalse(Path(parent, name).is_symlink())

        bin_dir.add(self.source.path / "Inspiral")
        (self.source.path / "Machine.yaml").write_text("Machine: new\n")
        bin_dir.freeze(self.source)
        self.assertEqual(
            (bin_dir.path / "Machine.yaml").read_text(), "Machine: new\n"
        )
        self.assertFalse((bin_dir.path / "Inspiral").exists())

        (self.build_dir / "CMakeCache.txt").write_text(
            "BUILD_SHARED_LIBS:BOOL=ON\n"
        )
        with self.assertRaisesRegex(RuntimeError, "BUILD_SHARED_LIBS"):
            bin_dir.freeze(self.source)

    def test_link_and_add(self):
        root = self.segment("0000_Inspiral").freeze(self.source)
        first = self.segment("0001_Inspiral").link(root)
        self.assertEqual(os.readlink(first.path), "../0000_Inspiral/bin")
        self.assertEqual(first.path.resolve(), root.path)

        second = self.segment("0002_Ringdown").link(first)
        self.assertEqual(os.readlink(second.path), "../0000_Inspiral/bin")

        added = second.add(self.source.path / "Inspiral")
        self.assertEqual(added, root.path / "Inspiral")
        (self.source.path / "Inspiral").write_text("#!/bin/bash\n# rebuilt\n")
        self.assertEqual(second.add(self.source.path / "Inspiral"), added)
        self.assertNotIn("rebuilt", added.read_text())

        (self.build_dir / "CMakeCache.txt").write_text(
            "BUILD_SHARED_LIBS:BOOL=OFF\n"
        )
        linked_to_build = self.segment("0003_Inspiral").link(self.source)
        outside = self.test_dir / "Outside"
        _write_executable(outside)
        with self.assertRaisesRegex(RuntimeError, "not in the build directory"):
            linked_to_build.add(outside)
        self.assertFalse((self.source.path / "Outside").exists())


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
