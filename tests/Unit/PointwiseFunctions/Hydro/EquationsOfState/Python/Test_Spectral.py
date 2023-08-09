# Distributed under the MIT License.
# See LICENSE.txt for details.


import unittest

import numpy as np
import numpy.testing as npt

import spectre.PointwiseFunctions.Hydro.EquationsOfState as spectre_eos
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar


class TestSpectral(unittest.TestCase):
    def test_spectral(self):
        # Parameters are sLy$\Gamma$2 from https://doi.org/10.48550/arXiv.1908.05277

        eos = spectre_eos.Spectral(
            reference_density=1.0118e-4,
            reference_pressure=3.33625e-7,
            spectral_coefficients=[2, 0, 0.4029, -0.1008],
            upper_density=6e-3,
        )

        # These are the pressures calculated in python implementation
        # testing the bindings. The python implementation will be added
        # in a second merge request.

        pressures_pyth = [
            3.258886510983e-7,
            7.846581912375e-7,
            1.990840482194e-6,
            5.508910104189e-6,
            1.683916561373e-5,
            5.636080617205e-5,
            2.003712243313e-4,
            7.183209159534e-4,
            2.412536276978e-3,
            6.901956906998e-3,
        ]
        self.assertTrue(
            isinstance(eos, spectre_eos.RelativisticEquationOfState1D)
        )
        # Only testing p(rho) because the class is tested thoroughly in C++
        density = np.geomspace(1e-4, 5e-3, 10)
        pressure = eos.pressure_from_density(Scalar[DataVector](density))

        npt.assert_allclose(
            pressure,
            [pressures_pyth],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
