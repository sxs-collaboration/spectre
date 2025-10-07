# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators.TimeDependentOptions import (
    BinaryCompactObjectTimeDependentOptions,
    ExpansionMapOptions,
    KerrSchildFromBoyerLindquist,
    RotationMapOptions,
    ShapeMapOptions,
    SkewMapOptions,
    TranslationMapOptions,
)


class TestBinaryCompactObjectTimeDependentOptions(unittest.TestCase):
    def test_construction(self):
        expansion_map = ExpansionMapOptions([1.0, 1e-4, 0.0], 100.0, 1e-6)
        rotation_map = RotationMapOptions([[0.0, 0.0, 0.0, 1.0]], 100.0)
        translation_map = TranslationMapOptions(
            [[1.0, -1.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        shape_map_a = ShapeMapOptions["A"](
            10, KerrSchildFromBoyerLindquist(0.66, [0.0, 0.0, 0.99]), None
        )
        shape_map_b = ShapeMapOptions["B"](
            10, KerrSchildFromBoyerLindquist(0.33, [0.0, 0.0, -0.99]), None
        )
        skew_map = SkewMapOptions([0.0, 40.0, 50.0], [0.0, 10.0, 0.0])
        binary_compact_object_one_map = BinaryCompactObjectTimeDependentOptions(
            0.0,
            None,
            None,
            translation_map,
            None,
            None,
            None,
            None,
        )
        binary_compact_object_all_maps = (
            BinaryCompactObjectTimeDependentOptions(
                0.0,
                expansion_map,
                rotation_map,
                translation_map,
                skew_map,
                shape_map_a,
                shape_map_b,
                None,
            )
        )
        self.assertNotEqual(binary_compact_object_one_map, None)
        self.assertNotEqual(binary_compact_object_all_maps, None)
        self.assertNotEqual(
            binary_compact_object_one_map, binary_compact_object_all_maps
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
