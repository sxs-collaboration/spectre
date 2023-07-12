# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

import spectre.PointwiseFunctions.GeneralRelativity.GeneralizedHarmonic as gh
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr


class TestBindings(unittest.TestCase):
    def test_deriv_spatial_metric(self):
        phi = tnsr.iaa[DataVector, 3](num_points=1, fill=1.0)
        gh.deriv_spatial_metric(
            phi,
        )

    def test_time_deriv_of_shift(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=0.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        spacetime_unit_normal = tnsr.A[DataVector, 3](num_points=1, fill=1.0)
        phi = tnsr.iaa[DataVector, 3](num_points=1, fill=1.0)
        pi = tnsr.aa[DataVector, 3](num_points=1, fill=1.0)
        gh.time_deriv_of_shift(
            lapse,
            shift,
            inverse_spatial_metric,
            spacetime_unit_normal,
            phi,
            pi,
        )

    def test_spatial_ricci_tensor(self):
        phi = tnsr.iaa[DataVector, 3](num_points=1, fill=0.0)
        deriv_phi = tnsr.ijaa[DataVector, 3](num_points=1, fill=0.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=0.0)
        gh.spatial_ricci_tensor(
            phi,
            deriv_phi,
            inverse_spatial_metric,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
