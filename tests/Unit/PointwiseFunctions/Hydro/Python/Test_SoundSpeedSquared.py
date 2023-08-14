# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import sys
import unittest

import numpy as np
import numpy.testing as npt

import spectre.PointwiseFunctions.Hydro as hydro
import spectre.PointwiseFunctions.Hydro.EquationsOfState as eos
from spectre import Informer
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TestFunctions import *


class TestSoundSpeedSquared(unittest.TestCase):
    def test_sound_speed_squared_1d(self):
        rest_mass_density = Scalar[DataVector](num_points=5, fill=2.1)
        specific_internal_energy = Scalar[DataVector](num_points=5, fill=1.1)
        specific_enthalpy = Scalar[DataVector](num_points=5, fill=1.8)
        equation_of_state = eos.RelativisticPolytropicFluid(0.003, 4.0 / 3.0)

        bindings = hydro.sound_speed_squared(
            rest_mass_density,
            specific_internal_energy,
            specific_enthalpy,
            equation_of_state,
        )

        alternative = np.array(
            (
                equation_of_state.chi_from_density(np.array(rest_mass_density))[
                    0
                ]
                + equation_of_state.kappa_times_p_over_rho_squared_from_density(
                    np.array(rest_mass_density)
                )[0]
            )
            / np.array(specific_enthalpy)[0]
        )[0]

        assert type(bindings) == Scalar[DataVector]
        np.testing.assert_allclose(bindings, alternative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
