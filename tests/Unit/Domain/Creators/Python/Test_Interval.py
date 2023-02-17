# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Domain.Creators import Interval, DomainCreator1D
from spectre.Domain.CoordinateMaps import Distribution
import unittest


class TestInterval(unittest.TestCase):
    def test_construction(self):
        interval = Interval(lower_x=[1.],
                            upper_x=[2.],
                            distribution=Distribution.Linear,
                            singularity=None,
                            is_periodic_in_x=[False],
                            initial_refinement_level_x=[1],
                            initial_number_of_grid_points_in_x=[3])
        self.assertIsInstance(interval, DomainCreator1D)


if __name__ == '__main__':
    unittest.main(verbosity=2)
