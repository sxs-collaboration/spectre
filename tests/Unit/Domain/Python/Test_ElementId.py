# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain import ElementId1D, ElementId2D, ElementId3D, SegmentId
import unittest


class TestElementId(unittest.TestCase):
    def test_construction(self):
        element_id = ElementId1D(block_id=1)
        self.assertEqual(element_id.block_id, 1)
        self.assertEqual(element_id.segment_ids, [SegmentId(0, 0)])
        element_id = ElementId1D(block_id=1, segment_ids=[SegmentId(1, 0)])
        self.assertEqual(element_id.block_id, 1)
        self.assertEqual(element_id.segment_ids, [SegmentId(1, 0)])

    def test_repr(self):
        self.assertEqual(repr(ElementId1D(0)), "[B0,(L0I0)]")
        self.assertEqual(
            repr(ElementId2D(
                2, [SegmentId(0, 0), SegmentId(1, 0)])), "[B2,(L0I0,L1I0)]")
        self.assertEqual(
            repr(
                ElementId3D(
                    1, [SegmentId(1, 0),
                        SegmentId(0, 0),
                        SegmentId(1, 1)])), "[B1,(L1I0,L0I0,L1I1)]")

    def test_equality(self):
        self.assertEqual(ElementId1D(0, [SegmentId(1, 0)]),
                         ElementId1D(0, [SegmentId(1, 0)]))
        self.assertNotEqual(ElementId1D(0, [SegmentId(1, 0)]),
                            ElementId1D(0, [SegmentId(1, 1)]))

    def test_external_boundary_id(self):
        self.assertEqual(ElementId1D.external_boundary_id(),
                         ElementId1D.external_boundary_id())


if __name__ == '__main__':
    unittest.main(verbosity=2)
