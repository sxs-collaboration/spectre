# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain.Creators import Sphere, DomainCreator3D
import unittest


class TestSphere(unittest.TestCase):
    def test_construction(self):
        sphere = Sphere(
            inner_radius=1.0,
            outer_radius=2.0,
            inner_cube_sphericity=0.0,
            initial_refinement=1,
            initial_number_of_grid_points=3,
            use_equiangular_map=False,
        )
        self.assertIsInstance(sphere, DomainCreator3D)
        shell = Sphere(
            inner_radius=1.0,
            outer_radius=2.0,
            excise=True,
            initial_refinement=1,
            initial_number_of_grid_points=3,
            use_equiangular_map=True,
        )
        self.assertIsInstance(shell, DomainCreator3D)


if __name__ == "__main__":
    unittest.main(verbosity=2)
