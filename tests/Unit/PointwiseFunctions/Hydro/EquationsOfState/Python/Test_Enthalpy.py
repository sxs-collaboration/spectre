# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

import spectre
import spectre.PointwiseFunctions.Hydro.EquationsOfState as EoS
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar


class TestEnthalpy(unittest.TestCase):
    def test_enthalpy(self):
        # Parameters are DBHF parameters from https://doi.org/10.48550/arXiv.2301.13818

        low_eos_spectre = EoS.Spectral(
            reference_density=4.533804669935759e-05,
            reference_pressure=9.970647727158039e-08,
            spectral_coefficients=[
                1.2,
                0.0,
                1.34440187653529,
                -0.46098357752567365,
            ],
            upper_density=0.0004533804669935759,
        )

        eos_spectre = EoS.EnthalpySpectral(
            reference_density=0.00022669023349678794,
            max_density=0.0031736632689550316,
            min_density=0.0004533804669935759,
            trig_scale=1.26426988871305,
            polynomial_coefficients=[
                1.0,
                0.08063293075870805,
                4.406887319408924e-26,
                8.177895241388924e-22,
                0.013558242085066733,
                0.004117320982626606,
                9.757362504479485e-26,
                1.5646573325075753e-30,
                0.00016253964205058317,
            ],
            sin_coefficients=[
                0.0003763514388305583,
                0.017968749910748837,
                0.008140052979970034,
                -0.003067418379116628,
                -0.0008236601907322793,
            ],
            cos_coefficients=[
                -0.01080996024705052,
                -0.003421193490191067,
                0.012325774692378716,
                0.004367136076912163,
                -0.00020374276952538073,
            ],
            low_density_eos=low_eos_spectre,
            transition_delta_epsilon=0.0,
        )
        self.assertTrue(
            isinstance(eos_spectre, EoS.RelativisticEquationOfState1D)
        )
        # Only testing p(rho) because the class is tested thoroughly in C++
        density = np.geomspace(4.6e-4, 5e-3, 10)
        pressure = eos_spectre.pressure_from_density(
            Scalar[DataVector](density)
        )
        # These are the pressures calculated in python implementation
        # testing the bindings. The python implementation will be added
        # in a second merge request.

        pressures_pyth = [
            1.5302849088016283e-5,
            3.1441124536164583e-5,
            6.5907292284151015e-5,
            1.4033564857885284e-4,
            3.0098826697494873e-4,
            6.4603643776028635e-4,
            1.3821660234437354e-3,
            2.9420820328679361e-3,
            6.2206944563790453e-3,
            1.3007001237896111e-2,
        ]

        npt.assert_allclose(
            pressure[0],
            pressures_pyth,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
