# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar
from spectre.PointwiseFunctions.Punctures import adm_mass_integrand


class TestBindings(unittest.TestCase):
    def test_adm_mass_integrand(self):
        field = np.random.rand(1, 5)
        alpha = np.random.rand(1, 5)
        beta = np.random.rand(1, 5)
        result = adm_mass_integrand(field, alpha, beta)
        npt.assert_allclose(
            np.array(result),
            1.0 / (2.0 * np.pi) * beta * (alpha * (1.0 + field) + 1.0) ** (-7),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
