# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import sys
import unittest

import numpy as np
import numpy.testing as npt

import spectre.DataStructures.Tensor.Frame as fr
import spectre.PointwiseFunctions.Hydro as hydro
from spectre import Informer
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TestFunctions import *


class TestSpecificEnthalpy(unittest.TestCase):
    def test_relativistic_specific_enthalpy(self):
        rest_mass_density = Scalar[DataVector](num_points=5, fill=2.1)
        specific_internal_energy = Scalar[DataVector](num_points=5, fill=1.1)
        pressure = Scalar[DataVector](num_points=5, fill=3.1)

        bindings = hydro.relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure
        )
        alternative = relativistic_specific_enthalpy(
            np.array(rest_mass_density),
            np.array(specific_internal_energy),
            np.array(pressure),
        )
        assert type(bindings) == Scalar[DataVector]
        np.testing.assert_allclose(bindings, alternative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
