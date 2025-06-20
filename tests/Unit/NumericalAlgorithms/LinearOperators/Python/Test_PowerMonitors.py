# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np

from spectre.DataStructures import DataVector
from spectre.NumericalAlgorithms.LinearOperators import (
    absolute_truncation_error,
    convergence_rate_and_number_of_pile_up_modes,
    power_monitors,
    relative_truncation_error,
)
from spectre.Spectral import (
    Basis,
    Mesh1D,
    Mesh2D,
    Quadrature,
    logical_coordinates,
)
from spectre.SphericalHarmonics import Frame, Strahlkorper, power_monitor


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
    def test_truncation_error(self):
        mesh = Mesh1D(2, Basis.Legendre, Quadrature.GaussLobatto)
        logical_coords = np.array(logical_coordinates(mesh))[0]

        # Define the test function
        slope, offset = 0.1, 1.0
        test_data = slope * logical_coords + offset

        # For a linear function the slope and offset correspond to the power
        # monitor values
        # The weighted average of the highest modes is
        avg = np.log10(np.abs(slope)) * np.exp(-0.25) + np.log10(
            np.abs(offset)
        ) * np.exp(-0.25)
        avg = avg / (np.exp(-0.25) + np.exp(-0.25))
        expected_relative_truncation_error = np.power(10.0, avg)
        expected_absolute_truncation_error = (
            np.max(np.abs(test_data)) * expected_relative_truncation_error
        )

        # Test relative truncation_error
        rel_error = relative_truncation_error(test_data, mesh)
        np.testing.assert_allclose(
            rel_error, expected_relative_truncation_error, 1e-12, 1e-12
        )

        # Test absolute truncation_error
        abs_error = absolute_truncation_error(test_data, mesh)
        np.testing.assert_allclose(
            abs_error, expected_absolute_truncation_error, 1e-12, 1e-12
        )

    # Check that the convergence rate for a straight line is consistent with
    # the analytic expectation
    def test_convergence_rate_and_pile_up_modes(self):
        slope, offset = -0.4, 1.4
        modes = np.arange(0, 14, 1)
        filtered_modes = 3
        test_power_monitor = 10.0 ** (slope * modes + offset)
        rate = convergence_rate_and_number_of_pile_up_modes(
            test_power_monitor, filtered_modes
        )["convergence_rate"]
        np.testing.assert_allclose(rate, -slope)

        # Check that a straight line with some pile up modes added by hand
        # recovers the correct integer number of pile up modes
        expected_pile_up_modes = 5
        modes_to_fill = filtered_modes + expected_pile_up_modes
        for i in range(0, modes_to_fill, 1):
            test_power_monitor[len(test_power_monitor) - i - 1] = (
                test_power_monitor[len(test_power_monitor) - modes_to_fill - 1]
            )
        pile_up_modes = convergence_rate_and_number_of_pile_up_modes(
            test_power_monitor, filtered_modes
        )["number_of_pile_up_modes"]
        np.testing.assert_allclose(
            np.floor(pile_up_modes), expected_pile_up_modes
        )


if __name__ == "__main__":
    unittest.main()
