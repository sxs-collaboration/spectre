# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators import DomainCreator3D, SphericalShells


class TestSphericalShells(unittest.TestCase):
    def test_construction(self):
        spherical_shells = SphericalShells(
            inner_radius=1.0,
            outer_radius=2.0,
            initial_radial_refinement=0,
            initial_number_of_radial_grid_points=4,
            initial_spherical_harmonic_l=8,
        )
        self.assertIsInstance(spherical_shells, DomainCreator3D)
        self.assertEqual(spherical_shells.create_domain().dim, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
