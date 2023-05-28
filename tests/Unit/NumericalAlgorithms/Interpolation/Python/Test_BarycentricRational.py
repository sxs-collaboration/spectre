# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt
import scipy.interpolate

from spectre.Interpolation import BarycentricRational


class TestBarycentricRational(unittest.TestCase):
    def test_barycentric_rational(self):
        x_values = np.linspace(0, 10, 5)
        y_values = x_values**2
        interpolant = BarycentricRational(x_values, y_values, order=3)
        npt.assert_array_equal(interpolant.x_values(), x_values)
        npt.assert_array_equal(interpolant.y_values(), y_values)
        self.assertEqual(interpolant.order(), 3)
        # Compare to scipy implementation
        scipy_interpolant = scipy.interpolate.BarycentricInterpolator(
            x_values, y_values
        )
        sample_points = [0.0, 1.5, 9.0, 10.0]
        for x in sample_points:
            self.assertAlmostEqual(interpolant(x), scipy_interpolant(x))


if __name__ == "__main__":
    unittest.main()
