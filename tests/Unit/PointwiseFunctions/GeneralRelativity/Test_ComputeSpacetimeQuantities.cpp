// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/range/combine.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_compute_1d_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto psi = gr::spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                        make_spatial_metric<dim>(0.));

  CHECK(psi.get(0, 0) == approx(-8.0));
  CHECK(psi.get(0, 1) == approx(1.0));
  CHECK(psi.get(1, 1) == approx(1.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::spacetime_metric(make_lapse(used_for_size),
                           make_shift<dim>(used_for_size),
                           make_spatial_metric<dim>(used_for_size)),
      psi);
}

void test_compute_2d_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto psi = gr::spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                        make_spatial_metric<dim>(0.));

  CHECK(psi.get(0, 0) == approx(16.0));
  CHECK(psi.get(0, 1) == approx(5.0));
  CHECK(psi.get(0, 2) == approx(10.0));
  CHECK(psi.get(1, 1) == approx(1.0));
  CHECK(psi.get(1, 2) == approx(2.0));
  CHECK(psi.get(2, 2) == approx(4.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::spacetime_metric(make_lapse(used_for_size),
                           make_shift<dim>(used_for_size),
                           make_spatial_metric<dim>(used_for_size)),
      psi);
}

void test_compute_3d_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto psi = gr::spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                        make_spatial_metric<dim>(0.));

  CHECK(psi.get(0, 0) == approx(187.0));
  CHECK(psi.get(0, 1) == approx(14.0));
  CHECK(psi.get(0, 2) == approx(28.0));
  CHECK(psi.get(0, 3) == approx(42.0));
  CHECK(psi.get(1, 1) == approx(1.0));
  CHECK(psi.get(1, 2) == approx(2.0));
  CHECK(psi.get(1, 3) == approx(3.0));
  CHECK(psi.get(2, 2) == approx(4.0));
  CHECK(psi.get(2, 3) == approx(6.0));
  CHECK(psi.get(3, 3) == approx(9.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::spacetime_metric(make_lapse(used_for_size),
                           make_shift<dim>(used_for_size),
                           make_spatial_metric<dim>(used_for_size)),
      psi);
}

void test_compute_1d_inverse_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                   make_inverse_spatial_metric<dim>(0.));

  CHECK(inverse_spacetime_metric.get(0, 0) == approx(-1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 1) == approx(1. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 1) == approx(8. / 9.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::inverse_spacetime_metric(
          make_lapse(used_for_size), make_shift<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      inverse_spacetime_metric);
}

void test_compute_2d_inverse_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                   make_inverse_spatial_metric<dim>(0.));

  CHECK(inverse_spacetime_metric.get(0, 0) == approx(-1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 1) == approx(1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 2) == approx(2. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 1) == approx(8. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 2) == approx(16. / 9.));
  CHECK(inverse_spacetime_metric.get(2, 2) == approx(32. / 9.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::inverse_spacetime_metric(
          make_lapse(used_for_size), make_shift<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      inverse_spacetime_metric);
}

void test_compute_3d_inverse_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                   make_inverse_spatial_metric<dim>(0.));

  CHECK(inverse_spacetime_metric.get(0, 0) == approx(-1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 1) == approx(1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 2) == approx(2. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 3) == approx(3. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 1) == approx(8. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 2) == approx(16. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 3) == approx(24. / 9.));
  CHECK(inverse_spacetime_metric.get(2, 2) == approx(32. / 9.));
  CHECK(inverse_spacetime_metric.get(2, 3) == approx(48. / 9.));
  CHECK(inverse_spacetime_metric.get(3, 3) == approx(8.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::inverse_spacetime_metric(
          make_lapse(used_for_size), make_shift<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      inverse_spacetime_metric);
}

void test_compute_1d_derivatives_spacetime_metric(
    const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto d_spacetime_metric = gr::derivatives_of_spacetime_metric(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(d_spacetime_metric.get(0, 0, 0) == approx(-28.0));
  CHECK(d_spacetime_metric.get(0, 0, 1) == approx(1.0));
  CHECK(d_spacetime_metric.get(0, 1, 1) == approx(0.0));
  CHECK(d_spacetime_metric.get(1, 0, 0) == approx(2.0));
  CHECK(d_spacetime_metric.get(1, 0, 1) == approx(10.0));
  CHECK(d_spacetime_metric.get(1, 1, 1) == approx(3.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::derivatives_of_spacetime_metric(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_deriv_lapse<dim>(used_for_size), make_shift<dim>(used_for_size),
          make_dt_shift<dim>(used_for_size),
          make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_dt_spatial_metric<dim>(used_for_size),
          make_deriv_spatial_metric<dim>(used_for_size)),
      d_spacetime_metric);
}

void test_compute_2d_derivatives_spacetime_metric(
    const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto d_spacetime_metric = gr::derivatives_of_spacetime_metric(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(d_spacetime_metric.get(0, 0, 0) == approx(82.0));
  CHECK(d_spacetime_metric.get(0, 0, 1) == approx(12.0));
  CHECK(d_spacetime_metric.get(0, 0, 2) == approx(25.0));
  CHECK(d_spacetime_metric.get(0, 1, 1) == approx(0.0));
  CHECK(d_spacetime_metric.get(0, 1, 2) == approx(1.0));
  CHECK(d_spacetime_metric.get(0, 2, 2) == approx(2.0));
  CHECK(d_spacetime_metric.get(1, 0, 0) == approx(330.0));
  CHECK(d_spacetime_metric.get(1, 0, 1) == approx(42.0));
  CHECK(d_spacetime_metric.get(1, 0, 2) == approx(84.0));
  CHECK(d_spacetime_metric.get(1, 1, 1) == approx(3.0));
  CHECK(d_spacetime_metric.get(1, 1, 2) == approx(6.0));
  CHECK(d_spacetime_metric.get(1, 2, 2) == approx(12.0));
  CHECK(d_spacetime_metric.get(2, 0, 0) == approx(294.0));
  CHECK(d_spacetime_metric.get(2, 0, 1) == approx(42.0));
  CHECK(d_spacetime_metric.get(2, 0, 2) == approx(81.0));
  CHECK(d_spacetime_metric.get(2, 1, 1) == approx(4.0));
  CHECK(d_spacetime_metric.get(2, 1, 2) == approx(7.0));
  CHECK(d_spacetime_metric.get(2, 2, 2) == approx(13.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::derivatives_of_spacetime_metric(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_deriv_lapse<dim>(used_for_size), make_shift<dim>(used_for_size),
          make_dt_shift<dim>(used_for_size),
          make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_dt_spatial_metric<dim>(used_for_size),
          make_deriv_spatial_metric<dim>(used_for_size)),
      d_spacetime_metric);
}

void test_compute_3d_derivatives_spacetime_metric(
    const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto d_spacetime_metric = gr::derivatives_of_spacetime_metric(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(d_spacetime_metric.get(0, 0, 0) == approx(1242.0));
  CHECK(d_spacetime_metric.get(0, 0, 1) == approx(50.0));
  CHECK(d_spacetime_metric.get(0, 0, 2) == approx(98.0));
  CHECK(d_spacetime_metric.get(0, 0, 3) == approx(146.0));
  CHECK(d_spacetime_metric.get(0, 1, 1) == approx(0.0));
  CHECK(d_spacetime_metric.get(0, 1, 2) == approx(1.0));
  CHECK(d_spacetime_metric.get(0, 1, 3) == approx(2.0));
  CHECK(d_spacetime_metric.get(0, 2, 2) == approx(2.0));
  CHECK(d_spacetime_metric.get(0, 2, 3) == approx(3.0));
  CHECK(d_spacetime_metric.get(0, 3, 3) == approx(4.0));
  CHECK(d_spacetime_metric.get(1, 0, 0) == approx(2421.0));
  CHECK(d_spacetime_metric.get(1, 0, 1) == approx(108.0));
  CHECK(d_spacetime_metric.get(1, 0, 2) == approx(216.0));
  CHECK(d_spacetime_metric.get(1, 0, 3) == approx(324.0));
  CHECK(d_spacetime_metric.get(1, 1, 1) == approx(3.0));
  CHECK(d_spacetime_metric.get(1, 1, 2) == approx(6.0));
  CHECK(d_spacetime_metric.get(1, 1, 3) == approx(9.0));
  CHECK(d_spacetime_metric.get(1, 2, 2) == approx(12.0));
  CHECK(d_spacetime_metric.get(1, 2, 3) == approx(18.0));
  CHECK(d_spacetime_metric.get(1, 3, 3) == approx(27.0));
  CHECK(d_spacetime_metric.get(2, 0, 0) == approx(2274.0));
  CHECK(d_spacetime_metric.get(2, 0, 1) == approx(108.0));
  CHECK(d_spacetime_metric.get(2, 0, 2) == approx(210.0));
  CHECK(d_spacetime_metric.get(2, 0, 3) == approx(312.0));
  CHECK(d_spacetime_metric.get(2, 1, 1) == approx(4.0));
  CHECK(d_spacetime_metric.get(2, 1, 2) == approx(7.0));
  CHECK(d_spacetime_metric.get(2, 1, 3) == approx(10.0));
  CHECK(d_spacetime_metric.get(2, 2, 2) == approx(13.0));
  CHECK(d_spacetime_metric.get(2, 2, 3) == approx(19.0));
  CHECK(d_spacetime_metric.get(2, 3, 3) == approx(28.0));
  CHECK(d_spacetime_metric.get(3, 0, 0) == approx(2127.0));
  CHECK(d_spacetime_metric.get(3, 0, 1) == approx(108.0));
  CHECK(d_spacetime_metric.get(3, 0, 2) == approx(204.0));
  CHECK(d_spacetime_metric.get(3, 0, 3) == approx(300.0));
  CHECK(d_spacetime_metric.get(3, 1, 1) == approx(5.0));
  CHECK(d_spacetime_metric.get(3, 1, 2) == approx(8.0));
  CHECK(d_spacetime_metric.get(3, 1, 3) == approx(11.0));
  CHECK(d_spacetime_metric.get(3, 2, 2) == approx(14.0));
  CHECK(d_spacetime_metric.get(3, 2, 3) == approx(20.0));
  CHECK(d_spacetime_metric.get(3, 3, 3) == approx(29.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::derivatives_of_spacetime_metric(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_deriv_lapse<dim>(used_for_size), make_shift<dim>(used_for_size),
          make_dt_shift<dim>(used_for_size),
          make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_dt_spatial_metric<dim>(used_for_size),
          make_deriv_spatial_metric<dim>(used_for_size)),
      d_spacetime_metric);
}

template <size_t Dim>
void test_compute_spacetime_normal_one_form(const DataVector& used_for_size) {
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<Dim, Frame::Inertial>(make_lapse(0.));

  CHECK(spacetime_normal_one_form.get(0) == approx(-3.0));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(spacetime_normal_one_form.get(i + 1) == 0.);
  }
  check_tensor_doubles_equals_tensor_datavectors(
      gr::spacetime_normal_one_form<Dim, Frame::Inertial>(
          make_lapse(used_for_size)),
      spacetime_normal_one_form);
}

void test_compute_1d_spacetime_normal_vector(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(make_lapse(0.), make_shift<dim>(0.));

  CHECK(spacetime_normal_vector.get(0) == approx(1. / 3.));
  CHECK(spacetime_normal_vector.get(1) == approx(-1. / 3.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::spacetime_normal_vector(make_lapse(used_for_size),
                                  make_shift<dim>(used_for_size)),
      spacetime_normal_vector);
}
void test_compute_2d_spacetime_normal_vector(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(make_lapse(0.), make_shift<dim>(0.));

  CHECK(spacetime_normal_vector.get(0) == approx(1. / 3.));
  CHECK(spacetime_normal_vector.get(1) == approx(-1. / 3.));
  CHECK(spacetime_normal_vector.get(2) == approx(-2. / 3.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::spacetime_normal_vector(make_lapse(used_for_size),
                                  make_shift<dim>(used_for_size)),
      spacetime_normal_vector);
}
void test_compute_3d_spacetime_normal_vector(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(make_lapse(0.), make_shift<dim>(0.));

  CHECK(spacetime_normal_vector.get(0) == approx(1. / 3.));
  CHECK(spacetime_normal_vector.get(1) == approx(-1. / 3.));
  CHECK(spacetime_normal_vector.get(2) == approx(-2. / 3.));
  CHECK(spacetime_normal_vector.get(3) == approx(-1.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::spacetime_normal_vector(make_lapse(used_for_size),
                                  make_shift<dim>(used_for_size)),
      spacetime_normal_vector);
}

void test_compute_1d_extrinsic_curvature(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto extrinsic_curvature = gr::extrinsic_curvature(
      make_lapse(0.), make_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(extrinsic_curvature.get(0, 0) == approx(17. / 6.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::extrinsic_curvature(make_lapse(used_for_size),
                              make_shift<dim>(used_for_size),
                              make_deriv_shift<dim>(used_for_size),
                              make_spatial_metric<dim>(used_for_size),
                              make_dt_spatial_metric<dim>(used_for_size),
                              make_deriv_spatial_metric<dim>(used_for_size)),
      extrinsic_curvature);
}
void test_compute_2d_extrinsic_curvature(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto extrinsic_curvature = gr::extrinsic_curvature(
      make_lapse(0.), make_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(extrinsic_curvature.get(0, 0) == approx(65. / 6.));
  CHECK(extrinsic_curvature.get(0, 1) == approx(97. / 6.));
  CHECK(extrinsic_curvature.get(1, 1) == approx(22.0));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::extrinsic_curvature(make_lapse(used_for_size),
                              make_shift<dim>(used_for_size),
                              make_deriv_shift<dim>(used_for_size),
                              make_spatial_metric<dim>(used_for_size),
                              make_dt_spatial_metric<dim>(used_for_size),
                              make_deriv_spatial_metric<dim>(used_for_size)),
      extrinsic_curvature);
}
void test_compute_3d_extrinsic_curvature(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto extrinsic_curvature = gr::extrinsic_curvature(
      make_lapse(0.), make_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(extrinsic_curvature.get(0, 0) == approx(79. / 3.));
  CHECK(extrinsic_curvature.get(0, 1) == approx(235. / 6.));
  CHECK(extrinsic_curvature.get(0, 2) == approx(52.0));
  CHECK(extrinsic_curvature.get(1, 1) == approx(53.0));
  CHECK(extrinsic_curvature.get(1, 2) == approx(2005. / 30.));
  CHECK(extrinsic_curvature.get(2, 2) == approx(245. / 3.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::extrinsic_curvature(make_lapse(used_for_size),
                              make_shift<dim>(used_for_size),
                              make_deriv_shift<dim>(used_for_size),
                              make_spatial_metric<dim>(used_for_size),
                              make_dt_spatial_metric<dim>(used_for_size),
                              make_deriv_spatial_metric<dim>(used_for_size)),
      extrinsic_curvature);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.SpacetimeDecomp",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv(2);
  test_compute_1d_spacetime_metric(dv);
  test_compute_2d_spacetime_metric(dv);
  test_compute_3d_spacetime_metric(dv);
  test_compute_1d_inverse_spacetime_metric(dv);
  test_compute_2d_inverse_spacetime_metric(dv);
  test_compute_3d_inverse_spacetime_metric(dv);
  test_compute_1d_derivatives_spacetime_metric(dv);
  test_compute_2d_derivatives_spacetime_metric(dv);
  test_compute_3d_derivatives_spacetime_metric(dv);
  test_compute_spacetime_normal_one_form<1>(dv);
  test_compute_spacetime_normal_one_form<2>(dv);
  test_compute_spacetime_normal_one_form<3>(dv);
  test_compute_1d_spacetime_normal_vector(dv);
  test_compute_2d_spacetime_normal_vector(dv);
  test_compute_3d_spacetime_normal_vector(dv);
  test_compute_1d_extrinsic_curvature(dv);
  test_compute_2d_extrinsic_curvature(dv);
  test_compute_3d_extrinsic_curvature(dv);
}
