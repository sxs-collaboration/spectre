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
  const auto psi = compute_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                            make_spatial_metric<dim>(0.));

  CHECK(psi.get(0, 0) == approx(-9.0));
  CHECK(psi.get(0, 1) == approx(0.0));
  CHECK(psi.get(1, 1) == approx(1.0));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_spacetime_metric(make_lapse(used_for_size),
                               make_shift<dim>(used_for_size),
                               make_spatial_metric<dim>(used_for_size)),
      psi);
}

void test_compute_2d_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto psi = compute_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                            make_spatial_metric<dim>(0.));

  CHECK(psi.get(0, 0) == approx(-5.0));
  CHECK(psi.get(0, 1) == approx(2.0));
  CHECK(psi.get(0, 2) == approx(4.0));
  CHECK(psi.get(1, 1) == approx(1.0));
  CHECK(psi.get(1, 2) == approx(2.0));
  CHECK(psi.get(2, 2) == approx(4.0));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_spacetime_metric(make_lapse(used_for_size),
                               make_shift<dim>(used_for_size),
                               make_spatial_metric<dim>(used_for_size)),
      psi);
}

void test_compute_3d_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto psi = compute_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                            make_spatial_metric<dim>(0.));
  CHECK(psi.get(0, 0) == approx(55.));
  CHECK(psi.get(0, 1) == approx(8.));
  CHECK(psi.get(0, 2) == approx(16.));
  CHECK(psi.get(0, 3) == approx(24.));
  CHECK(psi.get(1, 1) == approx(1.));
  CHECK(psi.get(1, 2) == approx(2.));
  CHECK(psi.get(1, 3) == approx(3.));
  CHECK(psi.get(2, 2) == approx(4.));
  CHECK(psi.get(2, 3) == approx(6.));
  CHECK(psi.get(3, 3) == approx(9.));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_spacetime_metric(make_lapse(used_for_size),
                               make_shift<dim>(used_for_size),
                               make_spatial_metric<dim>(used_for_size)),
      psi);
}

void test_compute_1d_inverse_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto inverse_spacetime_metric =
      compute_inverse_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                       make_inverse_spatial_metric<dim>(0.));
  CHECK(inverse_spacetime_metric.get(0, 0) == approx(-1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 1) == approx(0.0));
  CHECK(inverse_spacetime_metric.get(1, 1) == approx(1.0));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_inverse_spacetime_metric(
          make_lapse(used_for_size), make_shift<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      inverse_spacetime_metric);
}

void test_compute_2d_inverse_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 2;
  const auto inverse_spacetime_metric =
      compute_inverse_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                       make_inverse_spatial_metric<dim>(0.));
  CHECK(inverse_spacetime_metric.get(0, 0) == approx(-1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 1) == approx(0.0));
  CHECK(inverse_spacetime_metric.get(0, 2) == approx(1. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 1) == approx(1.0));
  CHECK(inverse_spacetime_metric.get(1, 2) == approx(2.0));
  CHECK(inverse_spacetime_metric.get(2, 2) == approx(35. / 9.));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_inverse_spacetime_metric(
          make_lapse(used_for_size), make_shift<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      inverse_spacetime_metric);
}

void test_compute_3d_inverse_spacetime_metric(const DataVector& used_for_size) {
  const size_t dim = 3;
  const auto inverse_spacetime_metric =
      compute_inverse_spacetime_metric(make_lapse(0.), make_shift<dim>(0.),
                                       make_inverse_spatial_metric<dim>(0.));
  CHECK(inverse_spacetime_metric.get(0, 0) == approx(-1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 1) == approx(0.));
  CHECK(inverse_spacetime_metric.get(0, 2) == approx(1. / 9.));
  CHECK(inverse_spacetime_metric.get(0, 3) == approx(2. / 9.));
  CHECK(inverse_spacetime_metric.get(1, 1) == approx(1.));
  CHECK(inverse_spacetime_metric.get(1, 2) == approx(2.));
  CHECK(inverse_spacetime_metric.get(1, 3) == approx(3.));
  CHECK(inverse_spacetime_metric.get(2, 2) == approx(35. / 9.));
  CHECK(inverse_spacetime_metric.get(2, 3) == approx(52. / 9.));
  CHECK(inverse_spacetime_metric.get(3, 3) == approx(77. / 9.));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_inverse_spacetime_metric(
          make_lapse(used_for_size), make_shift<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      inverse_spacetime_metric);
}

void test_compute_1d_derivatives_spacetime_metric(
    const DataVector& used_for_size) {
  const size_t dim = 1;
  const auto d_spacetime_metric = compute_derivatives_of_spacetime_metric(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(d_spacetime_metric.get(0, 0, 0) == approx(-30.0));
  CHECK(d_spacetime_metric.get(0, 0, 1) == approx(0.0));
  CHECK(d_spacetime_metric.get(0, 1, 1) == approx(0.0));
  CHECK(d_spacetime_metric.get(1, 0, 0) == approx(0.0));
  CHECK(d_spacetime_metric.get(1, 0, 1) == approx(3.0));
  CHECK(d_spacetime_metric.get(1, 1, 1) == approx(3.0));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_derivatives_of_spacetime_metric(
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
  const auto d_spacetime_metric = compute_derivatives_of_spacetime_metric(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(d_spacetime_metric.get(0, 0, 0) == approx(-12.0));
  CHECK(d_spacetime_metric.get(0, 0, 1) == approx(5.0));
  CHECK(d_spacetime_metric.get(0, 0, 2) == approx(10.0));
  CHECK(d_spacetime_metric.get(0, 1, 1) == approx(0.0));
  CHECK(d_spacetime_metric.get(0, 1, 2) == approx(1.0));
  CHECK(d_spacetime_metric.get(0, 2, 2) == approx(2.0));
  CHECK(d_spacetime_metric.get(1, 0, 0) == approx(72.0));
  CHECK(d_spacetime_metric.get(1, 0, 1) == approx(21.0));
  CHECK(d_spacetime_metric.get(1, 0, 2) == approx(42.0));
  CHECK(d_spacetime_metric.get(1, 1, 1) == approx(3.0));
  CHECK(d_spacetime_metric.get(1, 1, 2) == approx(6.0));
  CHECK(d_spacetime_metric.get(1, 2, 2) == approx(12.0));
  CHECK(d_spacetime_metric.get(2, 0, 0) == approx(46.0));
  CHECK(d_spacetime_metric.get(2, 0, 1) == approx(19.0));
  CHECK(d_spacetime_metric.get(2, 0, 2) == approx(37.0));
  CHECK(d_spacetime_metric.get(2, 1, 1) == approx(4.0));
  CHECK(d_spacetime_metric.get(2, 1, 2) == approx(7.0));
  CHECK(d_spacetime_metric.get(2, 2, 2) == approx(13.0));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_derivatives_of_spacetime_metric(
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
  const auto d_spacetime_metric = compute_derivatives_of_spacetime_metric(
      make_lapse(0.), make_dt_lapse(0.), make_deriv_lapse<dim>(0.),
      make_shift<dim>(0.), make_dt_shift<dim>(0.), make_deriv_shift<dim>(0.),
      make_spatial_metric<dim>(0.), make_dt_spatial_metric<dim>(0.),
      make_deriv_spatial_metric<dim>(0.));

  CHECK(d_spacetime_metric.get(0, 0, 0) == approx(384.));
  CHECK(d_spacetime_metric.get(0, 0, 1) == approx(29.));
  CHECK(d_spacetime_metric.get(0, 0, 2) == approx(56.));
  CHECK(d_spacetime_metric.get(0, 0, 3) == approx(83.));
  CHECK(d_spacetime_metric.get(0, 1, 1) == approx(0.));
  CHECK(d_spacetime_metric.get(0, 1, 2) == approx(1.));
  CHECK(d_spacetime_metric.get(0, 1, 3) == approx(2.));
  CHECK(d_spacetime_metric.get(0, 2, 2) == approx(2.));
  CHECK(d_spacetime_metric.get(0, 2, 3) == approx(3.));
  CHECK(d_spacetime_metric.get(0, 3, 3) == approx(4.));
  CHECK(d_spacetime_metric.get(1, 0, 0) == approx(864.));
  CHECK(d_spacetime_metric.get(1, 0, 1) == approx(66.));
  CHECK(d_spacetime_metric.get(1, 0, 2) == approx(132.));
  CHECK(d_spacetime_metric.get(1, 0, 3) == approx(198.));
  CHECK(d_spacetime_metric.get(1, 1, 1) == approx(3.));
  CHECK(d_spacetime_metric.get(1, 1, 2) == approx(6.));
  CHECK(d_spacetime_metric.get(1, 1, 3) == approx(9.));
  CHECK(d_spacetime_metric.get(1, 2, 2) == approx(12));
  CHECK(d_spacetime_metric.get(1, 2, 3) == approx(18));
  CHECK(d_spacetime_metric.get(1, 3, 3) == approx(27));
  CHECK(d_spacetime_metric.get(2, 0, 0) == approx(762.));
  CHECK(d_spacetime_metric.get(2, 0, 1) == approx(63.));
  CHECK(d_spacetime_metric.get(2, 0, 2) == approx(123.));
  CHECK(d_spacetime_metric.get(2, 0, 3) == approx(183.));
  CHECK(d_spacetime_metric.get(2, 1, 1) == approx(4.));
  CHECK(d_spacetime_metric.get(2, 1, 2) == approx(7.));
  CHECK(d_spacetime_metric.get(2, 1, 3) == approx(10));
  CHECK(d_spacetime_metric.get(2, 2, 2) == approx(13));
  CHECK(d_spacetime_metric.get(2, 2, 3) == approx(19));
  CHECK(d_spacetime_metric.get(2, 3, 3) == approx(28));
  CHECK(d_spacetime_metric.get(3, 0, 0) == approx(660.));
  CHECK(d_spacetime_metric.get(3, 0, 1) == approx(60.));
  CHECK(d_spacetime_metric.get(3, 0, 2) == approx(114.));
  CHECK(d_spacetime_metric.get(3, 0, 3) == approx(168.));
  CHECK(d_spacetime_metric.get(3, 1, 1) == approx(5.));
  CHECK(d_spacetime_metric.get(3, 1, 2) == approx(8.));
  CHECK(d_spacetime_metric.get(3, 1, 3) == approx(11.));
  CHECK(d_spacetime_metric.get(3, 2, 2) == approx(14.));
  CHECK(d_spacetime_metric.get(3, 2, 3) == approx(20.));
  CHECK(d_spacetime_metric.get(3, 3, 3) == approx(29.));

  check_tensor_doubles_equals_tensor_datavectors(
      compute_derivatives_of_spacetime_metric(
          make_lapse(used_for_size), make_dt_lapse(used_for_size),
          make_deriv_lapse<dim>(used_for_size), make_shift<dim>(used_for_size),
          make_dt_shift<dim>(used_for_size),
          make_deriv_shift<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size),
          make_dt_spatial_metric<dim>(used_for_size),
          make_deriv_spatial_metric<dim>(used_for_size)),
      d_spacetime_metric);
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
}
