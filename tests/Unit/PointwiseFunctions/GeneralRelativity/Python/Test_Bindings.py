# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.PointwiseFunctions.GeneralRelativity import psi4real
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr, Frame

import numpy as np
import numpy.testing as npt
import unittest


class TestPsi4(unittest.TestCase):
    def test_psi4(self):
        spatial_ricci = tnsr.ii[DataVector, 3](num_points=1, fill=0.)
        extrinsic_curvature = tnsr.ii[DataVector, 3](num_points=1, fill=0.)
        cov_deriv_extrinsic_curvature = tnsr.ijj[DataVector, 3](num_points=1,
                                                                fill=0.)
        flat_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.)
        flat_metric[0] = flat_metric[3] = flat_metric[5] = DataVector(1, 1.0)
        inverse_metric = tnsr.II[DataVector, 3](num_points=1, fill=0.)
        inverse_metric[0] = inverse_metric[3] = inverse_metric[5] = DataVector(
            1, 1.0)
        inertial_coords = tnsr.I[DataVector, 3](num_points=1, fill=1.)
        psi_4_real = psi4real(spatial_ricci, extrinsic_curvature,
                              cov_deriv_extrinsic_curvature, flat_metric,
                              inverse_metric, inertial_coords)
        npt.assert_allclose(psi_4_real, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
