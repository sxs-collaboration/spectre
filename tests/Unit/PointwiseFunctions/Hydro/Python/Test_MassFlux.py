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


class TestMassFlux(unittest.TestCase):
    def test_mass_flux(self):
        rest_mass_density = Scalar[DataVector](num_points=5, fill=2.1)
        spatial_velocity = tnsr.I[DataVector, 3, fr.Grid](
            num_points=5, fill=-1.0
        )

        lorentz_factor = Scalar[DataVector](num_points=5, fill=2.1)
        shift = tnsr.I[DataVector, 3, fr.Grid](num_points=5, fill=1.5)
        lapse = Scalar[DataVector](num_points=5, fill=1.1)
        sqrt_det_spatial_metric = Scalar[DataVector](num_points=5, fill=1.8)

        bindings = hydro.mass_flux(
            rest_mass_density,
            spatial_velocity,
            lorentz_factor,
            lapse,
            shift,
            sqrt_det_spatial_metric,
        )

        alternative = mass_flux(
            np.array(rest_mass_density)[0],
            np.array(spatial_velocity),
            np.array(lorentz_factor)[0],
            np.array(shift),
            np.array(lapse)[0],
            np.array(sqrt_det_spatial_metric)[0],
        )

        assert type(bindings) == tnsr.I[DataVector, 3, fr.Grid]
        np.testing.assert_allclose(bindings, alternative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
