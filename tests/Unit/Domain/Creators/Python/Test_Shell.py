# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain.Creators import Shell, DomainCreator3D
import unittest


class TestShell(unittest.TestCase):
    def test_construction(self):
        shell = Shell(inner_radius=1.,
                      outer_radius=2.,
                      initial_refinement=1,
                      initial_number_of_grid_points=[3, 3],
                      use_equiangular_map=False,
                      aspect_ratio=1.,
                      use_logarithmic_map=False)
        self.assertIsInstance(shell, DomainCreator3D)


if __name__ == '__main__':
    unittest.main(verbosity=2)
