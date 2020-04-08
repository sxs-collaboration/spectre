# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain.Creators import Rectangle, DomainCreator2D
import unittest


class TestRectangle(unittest.TestCase):
    def test_construction(self):
        rectangle = Rectangle(lower_xy=[1., 0.],
                              upper_xy=[2., 1.],
                              is_periodic_in_xy=[False, False],
                              initial_refinement_level_xy=[1, 0],
                              initial_number_of_grid_points_in_xy=[3, 4])
        self.assertIsInstance(rectangle, DomainCreator2D)


if __name__ == '__main__':
    unittest.main(verbosity=2)
