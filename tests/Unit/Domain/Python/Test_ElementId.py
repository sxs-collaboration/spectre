# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain import ElementId, SegmentId


class TestElementId(unittest.TestCase):
    def test_construction(self):
        element_id = ElementId[1](block_id=1)
        self.assertEqual(element_id.block_id, 1)
        self.assertEqual(element_id.segment_ids, [SegmentId(0, 0)])
        element_id = ElementId[1](block_id=1, segment_ids=[SegmentId(1, 0)])
        self.assertEqual(element_id.block_id, 1)
        self.assertEqual(element_id.segment_ids, [SegmentId(1, 0)])
        self.assertEqual(
            ElementId[2]("[B2,(L0I0,L2I3)]"),
            ElementId[2](
                block_id=2, segment_ids=[SegmentId(0, 0), SegmentId(2, 3)]
            ),
        )

    def test_repr(self):
        self.assertEqual(repr(ElementId[1](0)), "[B0,(L0I0)]")
        self.assertEqual(
            repr(ElementId[2](2, [SegmentId(0, 0), SegmentId(1, 0)])),
            "[B2,(L0I0,L1I0)]",
        )
        self.assertEqual(
            repr(
                ElementId[3](
                    1, [SegmentId(1, 0), SegmentId(0, 0), SegmentId(1, 1)]
                )
            ),
            "[B1,(L1I0,L0I0,L1I1)]",
        )

    def test_equality(self):
        self.assertEqual(
            ElementId[1](0, [SegmentId(1, 0)]),
            ElementId[1](0, [SegmentId(1, 0)]),
        )
        self.assertNotEqual(
            ElementId[1](0, [SegmentId(1, 0)]),
            ElementId[1](0, [SegmentId(1, 1)]),
        )

    def test_comparison(self):
        element1 = ElementId[1](0, [SegmentId(1, 0)])
        element2 = ElementId[1](0, [SegmentId(1, 1)])
        self.assertEqual(sorted([element2, element1]), [element1, element2])

    def test_hash(self):
        self.assertEqual(
            hash(ElementId[1](0, [SegmentId(1, 0)])),
            hash(ElementId[1](0, [SegmentId(1, 0)])),
        )
        self.assertNotEqual(
            hash(ElementId[1](0, [SegmentId(1, 0)])),
            hash(ElementId[1](0, [SegmentId(1, 1)])),
        )

    def test_external_boundary_id(self):
        self.assertEqual(
            ElementId[1].external_boundary_id(),
            ElementId[1].external_boundary_id(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
