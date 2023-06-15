# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr
from spectre.PointwiseFunctions.GeneralRelativity import *


class TestBindings(unittest.TestCase):
    def test_derivative_inverse_spatial_metric(self):
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        d_spatial_metric = tnsr.ijj[DataVector, 3](num_points=1, fill=0.0)
        deri_inverse_spatial_metric_tensor = deriv_inverse_spatial_metric(
            inverse_spatial_metric,
            d_spatial_metric,
        )

    def test_extrinsic_curvature(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=0.0)
        deriv_shift = tnsr.iJ[DataVector, 3](num_points=1, fill=0.0)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        spatial_metric[0] = spatial_metric[3] = spatial_metric[5] = DataVector(
            1, 1.0
        )
        dt_spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        deriv_spatial_metric = tnsr.ijj[DataVector, 3](num_points=1, fill=0.0)
        extrinsic_curv = extrinsic_curvature(
            lapse,
            shift,
            deriv_shift,
            spatial_metric,
            dt_spatial_metric,
            deriv_spatial_metric,
        )
        npt.assert_allclose(extrinsic_curv, 0)

    def test_inverse_spacetime_metric(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=-1.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=0.0)
        inverse_spacetime_metric_test = inverse_spacetime_metric(
            lapse, shift, inverse_spatial_metric
        )
        npt.assert_allclose(inverse_spacetime_metric_test, -1)

    def test_lapse_shift_normals(self):
        spacetime_metric = tnsr.aa[DataVector, 3](num_points=1, fill=1.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        shift_ = shift(spacetime_metric, inverse_spatial_metric)
        lapse_ = lapse(shift_, spacetime_metric)
        spacetime_normal_one_form(lapse_)
        spacetime_normal_vector(lapse_, shift_)

    def test_projections(self):
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        normal_vector = tnsr.I[DataVector, 3](num_points=1, fill=1.0)
        # tnsr.II
        transverse_projection_operator(inverse_spatial_metric, normal_vector)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=1.0)
        normal_one_form = tnsr.i[DataVector, 3](num_points=1, fill=1.0)
        # tnsr.ii
        transverse_projection_operator(spatial_metric, normal_one_form)
        # tnsr.Ij
        transverse_projection_operator(normal_vector, normal_one_form)
        spacetime_metric = tnsr.aa[DataVector, 3](num_points=1, fill=1.0)
        spacetime_normal_one_form = tnsr.a[DataVector, 3](
            num_points=1, fill=1.0
        )
        interface_unit_normal_one_form = tnsr.i[DataVector, 3](
            num_points=1, fill=1.0
        )
        # tnsr.aa
        transverse_projection_operator(
            spacetime_metric,
            spacetime_normal_one_form,
            interface_unit_normal_one_form,
        )
        inverse_spacetime_metric = tnsr.AA[DataVector, 3](
            num_points=1, fill=1.0
        )
        spacetime_normal_vector = tnsr.A[DataVector, 3](num_points=1, fill=1.0)
        interface_unit_normal_vector = tnsr.I[DataVector, 3](
            num_points=1, fill=1.0
        )
        # tnsr.AA
        transverse_projection_operator(
            inverse_spacetime_metric,
            spacetime_normal_vector,
            interface_unit_normal_vector,
        )
        # tnsr.Ab
        transverse_projection_operator(
            spacetime_normal_vector,
            spacetime_normal_one_form,
            interface_unit_normal_vector,
            interface_unit_normal_one_form,
        )

    def test_psi4(self):
        spatial_ricci = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        extrinsic_curvature = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        cov_deriv_extrinsic_curvature = tnsr.ijj[DataVector, 3](
            num_points=1, fill=0.0
        )
        flat_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        flat_metric[0] = flat_metric[3] = flat_metric[5] = DataVector(1, 1.0)
        inverse_metric = tnsr.II[DataVector, 3](num_points=1, fill=0.0)
        inverse_metric[0] = inverse_metric[3] = inverse_metric[5] = DataVector(
            1, 1.0
        )
        inertial_coords = tnsr.I[DataVector, 3](num_points=1, fill=1.0)
        psi_4_real = psi4real(
            spatial_ricci,
            extrinsic_curvature,
            cov_deriv_extrinsic_curvature,
            flat_metric,
            inverse_metric,
            inertial_coords,
        )
        npt.assert_allclose(psi_4_real, 0)

    def test_ricci(self):
        christoffel_second_kind = tnsr.Abb[DataVector, 3](
            num_points=1, fill=0.0
        )
        d_christoffel_second_kind = tnsr.aBcc[DataVector, 3](
            num_points=1, fill=0.0
        )
        spatial_ricci = ricci_tensor(
            christoffel_second_kind, d_christoffel_second_kind
        )
        inverse_metric = tnsr.AA[DataVector, 3](num_points=1, fill=0.0)
        ricci_scalar(spatial_ricci, inverse_metric)

    def test_derivatives_of_spacetime_metric(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.0)
        dt_lapse = Scalar[DataVector](num_points=1, fill=0.0)
        deriv_lapse = tnsr.i[DataVector, 3](num_points=1, fill=0.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=1.0)
        dt_shift = tnsr.I[DataVector, 3](num_points=1, fill=0.0)
        deriv_shift = tnsr.iJ[DataVector, 3](num_points=1, fill=0.0)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=1.0)
        dt_spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        deriv_spatial_metric = tnsr.ijj[DataVector, 3](num_points=1, fill=0.0)
        derivatives_of_spacetime_metric_test = derivatives_of_spacetime_metric(
            lapse,
            dt_lapse,
            deriv_lapse,
            shift,
            dt_shift,
            deriv_shift,
            spatial_metric,
            dt_spatial_metric,
            deriv_spatial_metric,
        )
        npt.assert_allclose(derivatives_of_spacetime_metric_test, 0)

    def test_spacetime_metric(self):
        lapse = Scalar[DataVector](num_points=1, fill=0.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=1.0)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        spacetime_metric_test = spacetime_metric(lapse, shift, spatial_metric)
        npt.assert_allclose(spacetime_metric_test, 0)

    def test_spatial_metric(self):
        spacetime_metric = tnsr.aa[DataVector, 3](num_points=1, fill=0.0)
        spatial_metric_test = spatial_metric(spacetime_metric)
        npt.assert_allclose(spatial_metric_test, 0)

    def test_time_derivative_of_spacetime_metric(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.0)
        dt_lapse = Scalar[DataVector](num_points=1, fill=0.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=1.0)
        dt_shift = tnsr.I[DataVector, 3](num_points=1, fill=0.0)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=1.0)
        dt_spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        t_deriv_of_spacetime_metric_test = time_derivative_of_spacetime_metric(
            lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric
        )
        npt.assert_allclose(t_deriv_of_spacetime_metric_test, 0)

    def test_time_derivative_of_spatial_metric(self):
        lapse = Scalar[DataVector](num_points=1, fill=1.0)
        shift = tnsr.I[DataVector, 3](num_points=1, fill=1.0)
        deriv_shift = tnsr.iJ[DataVector, 3](num_points=1, fill=0.0)
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=1.0)
        deriv_spatial_metric = tnsr.ijj[DataVector, 3](num_points=1, fill=0.0)
        extrinsic_curvature = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        t_deriv_of_spatial_metric_test = time_derivative_of_spatial_metric(
            lapse,
            shift,
            deriv_shift,
            spatial_metric,
            deriv_spatial_metric,
            extrinsic_curvature,
        )
        npt.assert_allclose(t_deriv_of_spatial_metric_test, 0)

    def test_weyl_electric(self):
        spatial_ricci = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        extrinsic_curvature = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        weyl_electric_tensor = weyl_electric(
            spatial_ricci, extrinsic_curvature, inverse_spatial_metric
        )
        weyl_electric_scalar(weyl_electric_tensor, inverse_spatial_metric)

    def test_weyl_propagating(self):
        ricci = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        extrinsic_curvature = tnsr.ii[DataVector, 3](num_points=1, fill=0.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        cov_deriv_extrinsic_curvature = tnsr.ijj[DataVector, 3](
            num_points=1, fill=0.0
        )
        unit_interface_normal_vector = tnsr.I[DataVector, 3](
            num_points=1, fill=1.0
        )
        projection_IJ = tnsr.II[DataVector, 3](num_points=1, fill=1.0)
        projection_ij = tnsr.ii[DataVector, 3](num_points=1, fill=1.0)
        projection_Ij = tnsr.Ij[DataVector, 3](num_points=1, fill=1.0)
        weyl_prop = weyl_propagating(
            ricci,
            extrinsic_curvature,
            inverse_spatial_metric,
            cov_deriv_extrinsic_curvature,
            unit_interface_normal_vector,
            projection_IJ,
            projection_ij,
            projection_Ij,
            sign=1.0,
        )
        npt.assert_allclose(weyl_prop, 0)

    def test_weyl_magnetic(self):
        grad_extrinsic_curvature = tnsr.ijj[DataVector, 3](
            num_points=1, fill=1.0
        )
        spatial_metric = tnsr.ii[DataVector, 3](num_points=1, fill=1.0)
        sqrt_det_spatial_metric = Scalar[DataVector](num_points=1, fill=1.0)
        weyl_mag = weyl_magnetic(
            grad_extrinsic_curvature, spatial_metric, sqrt_det_spatial_metric
        )
        npt.assert_allclose(weyl_mag, 0)

    def test_weyl_magnetic_scalar(self):
        weyl_magnetic = tnsr.ii[DataVector, 3](num_points=1, fill=1.0)
        inverse_spatial_metric = tnsr.II[DataVector, 3](num_points=1, fill=0.0)
        weyl_mag_scalar = weyl_magnetic_scalar(
            weyl_magnetic, inverse_spatial_metric
        )
        npt.assert_allclose(weyl_mag_scalar, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
