# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.CoordinateMaps import Distribution
from spectre.Domain.Creators import DomainCreator1D, Interval


class TestInterval(unittest.TestCase):
    def test_construction(self):
        interval = Interval(
            lower_bounds=[1.0],
            upper_bounds=[2.0],
            is_periodic=[False],
            initial_refinement_levels=[1],
            initial_num_points=[3],
        )
        self.assertIsInstance(interval, DomainCreator1D)


if __name__ == "__main__":
    unittest.main(verbosity=2)
