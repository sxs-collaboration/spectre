# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain.Creators import Cylinder, DomainCreator3D
import unittest


class TestCylinder(unittest.TestCase):
    def test_construction(self):
        cylinder = Cylinder(inner_radius=1.,
                            outer_radius=2.,
                            lower_bound=0.,
                            upper_bound=1.,
                            is_periodic_in_z=False,
                            initial_refinement=1,
                            initial_number_of_grid_points=[3, 4, 5],
                            use_equiangular_map=False)
        self.assertIsInstance(cylinder, DomainCreator3D)


if __name__ == '__main__':
    unittest.main(verbosity=2)
