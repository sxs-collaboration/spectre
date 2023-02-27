# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np

from spectre.DataStructures import DataVector
from spectre.NumericalAlgorithms.LinearOperators import (
    power_monitors,
    relative_truncation_error,
    truncation_error,
)
from spectre.Spectral import (
    Basis,
    Mesh1D,
    Mesh2D,
    Quadrature,
    logical_coordinates,
)


class TestPowerMonitors(unittest.TestCase):
    # Check the case for a constant function where the power monitors
    # should be given by the first basis function
    def test_power_monitors(self):
        num_points_per_dimension = 4

        extent = num_points_per_dimension
        basis = Basis.Legendre
        quadrature = Quadrature.GaussLobatto
        mesh = Mesh2D(extent, basis, quadrature)

        test_vec = np.ones(mesh.number_of_grid_points())

        test_array = power_monitors(test_vec, mesh)
        np_test_array = np.asarray(test_array)

        check_vec_0 = np.zeros(num_points_per_dimension)
        check_vec_0[0] = 1.0 / np.sqrt(num_points_per_dimension)

        check_vec_1 = np.zeros(num_points_per_dimension)
        check_vec_1[0] = 1.0 / np.sqrt(num_points_per_dimension)

        np_check_array = np.array([check_vec_0, check_vec_1])

        np.testing.assert_allclose(np_test_array, np_check_array, 1e-12, 1e-12)

    # Check that the truncation error for a straight line is consistent with the
    # analytic expectation
    def test_relative_truncation_error(self):
        num_points_per_dimension = 2

        extent = num_points_per_dimension
        basis = Basis.Legendre
        quadrature = Quadrature.GaussLobatto
        mesh = Mesh1D(extent, basis, quadrature)
        logical_coords = np.array(logical_coordinates(mesh))[0]

        # Define the test function
        slope, offset = 0.1, 1.0
        test_vec = slope * logical_coords + offset

        modes_all_dim = power_monitors(test_vec, mesh)
        modes = modes_all_dim[0]

        test_relative_truncation_error_exponent = relative_truncation_error(
            modes, extent
        )

        test_relative_truncation_error = np.power(
            10.0, -1.0 * test_relative_truncation_error_exponent
        )

        # For a linear function the slope and offset correspond to the power
        # monitor values
        # The weighted average of the highest modes is
        avg = np.log10(np.abs(slope)) * np.exp(-0.25) + np.log10(
            np.abs(offset)
        ) * np.exp(-0.25)
        avg = avg / (np.exp(-0.25) + np.exp(-0.25))
        expected_relative_truncation_error = np.power(10.0, avg)

        np.testing.assert_allclose(
            test_relative_truncation_error,
            expected_relative_truncation_error,
            1e-12,
            1e-12,
        )

        # Test absolute truncation_error
        test_truncation_error = np.asarray(
            truncation_error(modes_all_dim, test_vec)
        )

        expected_truncation_error = (
            np.max(np.abs(test_vec)) * expected_relative_truncation_error
        )

        np.testing.assert_allclose(
            test_truncation_error, expected_truncation_error, 1e-12, 1e-12
        )


if __name__ == "__main__":
    unittest.main()
