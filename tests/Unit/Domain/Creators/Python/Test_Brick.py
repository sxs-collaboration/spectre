# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain.Creators import Brick
import unittest


class TestBrick(unittest.TestCase):
    def test_construction(self):
        brick = Brick(lower_xyz=[1., 0., 2.],
                      upper_xyz=[2., 1., 3.],
                      is_periodic_in_xyz=[False, False, False],
                      initial_refinement_level_xyz=[1, 0, 1],
                      initial_number_of_grid_points_in_xyz=[3, 4, 2])
        domain = brick.create_domain()
        self.assertFalse(domain.is_time_dependent())
        self.assertEqual(brick.block_names(), [])
        self.assertEqual(brick.block_groups(), {})
        self.assertEqual(brick.initial_extents(), [[3, 4, 2]])
        self.assertEqual(brick.initial_refinement_levels(), [[1, 0, 1]])
        self.assertEqual(brick.functions_of_time(), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
