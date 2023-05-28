# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

import spectre.PointwiseFunctions.Hydro.EquationsOfState as spectre_eos
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar


class TestPolytropicFluid(unittest.TestCase):
    def test_polytropic_fluid(self):
        K = 100.0
        Gamma = 2.0
        eos = spectre_eos.RelativisticPolytropicFluid(
            polytropic_constant=K, polytropic_exponent=Gamma
        )
        self.assertTrue(
            isinstance(eos, spectre_eos.RelativisticEquationOfState1D)
        )
        # Only testing p(rho) because the class is tested thoroughly in C++
        density = np.random.random_sample((1, 5))
        pressure = eos.pressure_from_density(Scalar[DataVector](density))
        npt.assert_allclose(pressure, K * density**Gamma)


if __name__ == "__main__":
    unittest.main(verbosity=2)
