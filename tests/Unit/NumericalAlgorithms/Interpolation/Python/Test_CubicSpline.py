# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt
import scipy.interpolate

from spectre.Interpolation import CubicSpline


class TestCubicSpline(unittest.TestCase):
    def test_cubic_spline(self):
        x_values = np.linspace(0, 10, 5)
        y_values = x_values**2
        interpolant = CubicSpline(x_values, y_values)
        npt.assert_array_equal(interpolant.x_values(), x_values)
        npt.assert_array_equal(interpolant.y_values(), y_values)
        # Compare to scipy implementation
        scipy_interpolant = scipy.interpolate.CubicSpline(
            x_values, y_values, bc_type="natural"
        )
        sample_points = [0.0, 1.5, 9.0, 10.0]
        for x in sample_points:
            self.assertAlmostEqual(interpolant(x), scipy_interpolant(x))


if __name__ == "__main__":
    unittest.main()
