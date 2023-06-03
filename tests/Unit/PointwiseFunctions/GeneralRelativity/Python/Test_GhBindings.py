# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr
from spectre.PointwiseFunctions.GeneralRelativity.GeneralizedHarmonic import *


class TestBindings(unittest.TestCase):
    def test_time_deriv_of_shift(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=0.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        spacetime_unit_normal = tnsr.A[DataVector, 3](num_points=1, fill=1.0)
        phi_ = tnsr.iaa[DataVector, 3](num_points=1, fill=1.0)
        pi_ = tnsr.aa[DataVector, 3](num_points=1, fill=1.0)
        time_deriv_of_shift(
            lapse,
            shift,
            inverse_spatial_metric,
            spacetime_unit_normal,
            phi_,
            pi_,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
