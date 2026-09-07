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
from spectre.support.Logging import configure_logging


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
        # The first segment in an empty directory starts the numbering at zero
        self.assertIsNone(Segment.last(self.test_dir))
        first_segment = Segment.next(self.test_dir, label="Inspiral")
        self.assertEqual(
            first_segment,
            Segment(
                path=self.test_dir / "0000_Inspiral", id=0, label="Inspiral"
            ),
        )
        self.assertEqual(Segment.match(first_segment.path), first_segment)
        self.assertIsNone(Segment.match(self.test_dir / "NotASegment"))

        self.assertEqual(list_segments(self.test_dir), [])
        first_segment.path.mkdir()
        self.assertEqual(list_segments(self.test_dir), [first_segment])
        self.assertEqual(Segment.last(self.test_dir), first_segment)

        # The next segment continues the numbering, and can have a different
        # label because a pipeline can continue with another executable
        next_segment = Segment.next(self.test_dir, label="Ringdown")
        self.assertEqual(
            next_segment,
            Segment(
                path=self.test_dir / "0001_Ringdown", id=1, label="Ringdown"
            ),
        )
        next_segment.path.mkdir()
        self.assertEqual(
            list_segments(self.test_dir), [first_segment, next_segment]
        )
        self.assertEqual(Segment.last(self.test_dir), next_segment)

        # Checkpoints are stored in the segment
        self.assertEqual(
            next_segment.checkpoints_dir, next_segment.path / "Checkpoints"
        )
        self.assertEqual(next_segment.checkpoints, [])
        (next_segment.checkpoints_dir / "Checkpoint_0000").mkdir(parents=True)
        self.assertEqual(
            next_segment.checkpoints,
            [
                Checkpoint(
                    path=next_segment.checkpoints_dir / "Checkpoint_0000", id=0
                )
            ],
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
