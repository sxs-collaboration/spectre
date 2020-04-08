# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain.Creators import Brick, DomainCreator3D
import unittest


class TestBrick(unittest.TestCase):
    def test_construction(self):
        brick = Brick(lower_xyz=[1., 0., 2.],
                      upper_xyz=[2., 1., 3.],
                      is_periodic_in_xyz=[False, False, False],
                      initial_refinement_level_xyz=[1, 0, 1],
                      initial_number_of_grid_points_in_xyz=[3, 4, 2])
        self.assertIsInstance(brick, DomainCreator3D)


if __name__ == '__main__':
    unittest.main(verbosity=2)
