# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain import ElementId, SegmentId, Side


class TestElementId(unittest.TestCase):
    def test_construction(self):
        element_id = ElementId[1](block_id=1)
        self.assertEqual(element_id.block_id, 1)
        self.assertEqual(element_id.grid_index, 0)
        self.assertEqual(element_id.refinement_levels, [0])
        self.assertEqual(element_id.segment_ids, [SegmentId(0, 0)])
        element_id = ElementId[1](block_id=1, segment_ids=[SegmentId(1, 0)])
        self.assertEqual(element_id.block_id, 1)
        self.assertEqual(element_id.grid_index, 0)
        self.assertEqual(element_id.refinement_levels, [1])
        self.assertEqual(element_id.segment_ids, [SegmentId(1, 0)])
        element_id = ElementId[2](
            block_id=2, segment_ids=[SegmentId(0, 0), SegmentId(2, 3)]
        )
        self.assertEqual(element_id.block_id, 2)
        self.assertEqual(element_id.grid_index, 0)
        self.assertEqual(element_id.refinement_levels, [0, 2])
        self.assertEqual(ElementId[2]("[B2,(L0I0,L2I3)]"), element_id)

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

    def test_members(self):
        parent = ElementId[1](0, [SegmentId(1, 0)])
        child_lower = ElementId[1](0, [SegmentId(2, 0)])
        child_upper = ElementId[1](0, [SegmentId(2, 1)])
        self.assertEqual(
            parent.id_of_child(dim=0, side=Side.Lower), child_lower
        )
        self.assertEqual(
            parent.id_of_child(dim=0, side=Side.Upper), child_upper
        )
        self.assertEqual(child_lower.id_of_parent(dim=0), parent)
        self.assertEqual(child_upper.id_of_parent(dim=0), parent)

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
