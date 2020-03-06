# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain import SegmentId
import unittest


class TestSegmentId(unittest.TestCase):
    def test_construction(self):
        segment_id = SegmentId(refinement_level=1, index=0)
        self.assertEqual(segment_id.refinement_level, 1)
        self.assertEqual(segment_id.index, 0)

    def test_repr(self):
        self.assertEqual(repr(SegmentId(1, 0)), "L1I0")

    def test_equality(self):
        self.assertEqual(SegmentId(1, 0), SegmentId(1, 0))
        self.assertNotEqual(SegmentId(1, 0), SegmentId(1, 1))
        self.assertNotEqual(SegmentId(2, 0), SegmentId(1, 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
