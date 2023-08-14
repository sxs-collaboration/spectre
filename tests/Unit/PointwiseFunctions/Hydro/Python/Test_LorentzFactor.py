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


class TestLorentzFactor(unittest.TestCase):
    def test_lorentz_factor(self):
        spatial_velocity = tnsr.I[DataVector, 3, fr.Grid](
            num_points=5, fill=-1.0
        )

        spatial_velocity_form = tnsr.i[DataVector, 3, fr.Grid](
            num_points=5, fill=1 / np.sqrt(3)
        )

        bindings = hydro.lorentz_factor(spatial_velocity, spatial_velocity_form)
        alternative = [
            lorentz_factor(
                np.array(spatial_velocity), np.array(spatial_velocity_form)
            )[0]
        ]
        assert type(bindings) == Scalar[DataVector]
        np.testing.assert_allclose(bindings, alternative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
