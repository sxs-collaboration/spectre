# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

from spectre.Informer import unit_test_build_path
from spectre.support.DirectoryStructure import (
    Checkpoint,
    Segment,
    list_checkpoints,
    list_segments,
)


class TestDirectoryStructure(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/DirectoryStructure"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_checkpoints(self):
        checkpoint = Checkpoint.match(self.test_dir / "Checkpoint_0002")
        self.assertEqual(
            checkpoint, Checkpoint(path=self.test_dir / "Checkpoint_0002", id=2)
        )

        self.assertEqual(list_checkpoints(self.test_dir), [])
        checkpoint.path.mkdir()
        self.assertEqual(list_checkpoints(self.test_dir), [checkpoint])

    def test_segments(self):
        first_segment = Segment.first(self.test_dir)
        self.assertEqual(
            first_segment, Segment(path=self.test_dir / "Segment_0000", id=0)
        )
        self.assertEqual(Segment.match(first_segment.path), first_segment)

        segment = Segment.match(self.test_dir / "Segment_0003")
        self.assertEqual(
            segment, Segment(path=self.test_dir / "Segment_0003", id=3)
        )

        next_segment = segment.next
        self.assertEqual(next_segment.id, 4)
        self.assertEqual(next_segment.path.name, "Segment_0004")
        self.assertEqual(
            next_segment.path.resolve().parent, segment.path.resolve().parent
        )

        self.assertEqual(list_segments(self.test_dir), [])
        first_segment.path.mkdir()
        segment.path.mkdir()
        next_segment.path.mkdir()
        self.assertEqual(
            list_segments(self.test_dir), [first_segment, segment, next_segment]
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
