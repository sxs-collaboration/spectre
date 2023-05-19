# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.PointwiseFunctions.GeneralRelativity import *
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr, Scalar

import numpy as np
import numpy.testing as npt
import unittest


class TestBindings(unittest.TestCase):
    def test_extrinsic_curvature(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=0.)
        deriv_shift = tnsr.iJ[DataVector, 3](num_points=1, fill=0.)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.)
        spatial_metric[0] = spatial_metric[3] = spatial_metric[5] = DataVector(
            1, 1.0)
        dt_spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.)
        deriv_spatial_metric = tnsr.ijj[DataVector, 3](num_points=1, fill=0.)
        extrinsic_curv = extrinsic_curvature(lapse, shift, deriv_shift,
                                             spatial_metric, dt_spatial_metric,
                                             deriv_spatial_metric)
        npt.assert_allclose(extrinsic_curv, 0)

    def test_lapse_shift_normals(self):
        spacetime_metric = tnsr.aa[DataVector, 3](num_points=1, fill=1.)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.)
        shift_ = shift(spacetime_metric, inverse_spatial_metric)
        lapse_ = lapse(shift_, spacetime_metric)
        spacetime_normal_one_form(lapse_)
        spacetime_normal_vector(lapse_, shift_)

    def test_projections(self):
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.)
        normal_vector = tnsr.I[DataVector, 3](num_points=1, fill=1.)
        # tnsr.II
        transverse_projection_operator(inverse_spatial_metric, normal_vector)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=1.)
        normal_one_form = tnsr.i[DataVector, 3](num_points=1, fill=1.)
        # tnsr.ii
        transverse_projection_operator(spatial_metric, normal_one_form)
        # tnsr.Ij
        transverse_projection_operator(normal_vector, normal_one_form)
        spacetime_metric = tnsr.aa[DataVector, 3](num_points=1, fill=1.)
        spacetime_normal_one_form = tnsr.a[DataVector, 3](num_points=1,
                                                          fill=1.)
        interface_unit_normal_one_form = tnsr.i[DataVector, 3](num_points=1,
                                                               fill=1.)
        # tnsr.aa
        transverse_projection_operator(spacetime_metric,
                                       spacetime_normal_one_form,
                                       interface_unit_normal_one_form)
        inverse_spacetime_metric = tnsr.AA[DataVector, 3](num_points=1,
                                                          fill=1.)
        spacetime_normal_vector = tnsr.A[DataVector, 3](num_points=1, fill=1.)
        interface_unit_normal_vector = tnsr.I[DataVector, 3](num_points=1,
                                                             fill=1.)
        # tnsr.AA
        transverse_projection_operator(inverse_spacetime_metric,
                                       spacetime_normal_vector,
                                       interface_unit_normal_vector)
        # tnsr.Ab
        transverse_projection_operator(spacetime_normal_vector,
                                       spacetime_normal_one_form,
                                       interface_unit_normal_vector,
                                       interface_unit_normal_one_form)

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

    def test_ricci(self):
        christoffel_second_kind = tnsr.Abb[DataVector, 3](num_points=1,
                                                          fill=0.)
        d_christoffel_second_kind = tnsr.aBcc[DataVector, 3](num_points=1,
                                                             fill=0.)
        spatial_ricci = ricci_tensor(christoffel_second_kind,
                                     d_christoffel_second_kind)
        inverse_metric = tnsr.AA[DataVector, 3](num_points=1, fill=0.)
        ricci_scalar(spatial_ricci, inverse_metric)

    def test_weyl_propagating(self):
        ricci = tnsr.ii[DataVector, 3](num_points=1, fill=0.)
        extrinsic_curvature = tnsr.ii[DataVector, 3](num_points=1, fill=0.)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.)
        cov_deriv_extrinsic_curvature = tnsr.ijj[DataVector, 3](num_points=1,
                                                                fill=0.)
        unit_interface_normal_vector = tnsr.I[DataVector, 3](num_points=1,
                                                             fill=1.)
        projection_IJ = tnsr.II[DataVector, 3](num_points=1, fill=1.)
        projection_ij = tnsr.ii[DataVector, 3](num_points=1, fill=1.)
        projection_Ij = tnsr.Ij[DataVector, 3](num_points=1, fill=1.)
        weyl_prop = weyl_propagating(ricci,
                                     extrinsic_curvature,
                                     inverse_spatial_metric,
                                     cov_deriv_extrinsic_curvature,
                                     unit_interface_normal_vector,
                                     projection_IJ,
                                     projection_ij,
                                     projection_Ij,
                                     sign=1.0)
        npt.assert_allclose(weyl_prop, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
