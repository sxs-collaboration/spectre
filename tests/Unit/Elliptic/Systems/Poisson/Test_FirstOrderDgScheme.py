# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.TestHelpers.Elliptic.Systems.Poisson import (
    strong_first_order_dg_operator_matrix_1d)
import spectre.Domain.Creators as domain_creators
import spectre.DataStructures
import unittest
import numpy.testing as npt


class TestFirstOrderDgScheme(unittest.TestCase):
    def test_operator_matrix(self):
        domain = domain_creators.Interval(
            lower_x=[0.],
            upper_x=[1.],
            is_periodic_in_x=[False],
            initial_refinement_level_x=[1],
            initial_number_of_grid_points_in_x=[3])
        operator_matrix = strong_first_order_dg_operator_matrix_1d(domain)
        # We can't really test any properties of the operator without a useful
        # numerical flux. We can switch to the internal penalty numerical flux
        # once this PR is merged:
        # https://github.com/sxs-collaboration/spectre/pull/1725
        self.assertEqual(operator_matrix.shape, (12, 12))


if __name__ == '__main__':
    unittest.main(verbosity=2)
