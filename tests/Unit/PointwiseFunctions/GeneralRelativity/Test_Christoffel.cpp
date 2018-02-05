// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_1d_spatial_christoffel_first_kind(const DataVector &used_for_size) {
  const size_t spatial_dim = 1;
  const auto christoffel = gr::christoffel_first_kind(
      make_deriv_spatial_metric<spatial_dim>(0.));
  CHECK(christoffel.get(0, 0, 0) == approx(1.5));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::christoffel_first_kind(
          make_deriv_spatial_metric<spatial_dim>(used_for_size)),
      christoffel);
}

void test_2d_spatial_christoffel_first_kind(const DataVector &used_for_size) {
  const size_t spatial_dim = 2;
  const auto christoffel = gr::christoffel_first_kind(
      make_deriv_spatial_metric<spatial_dim>(0.));
  CHECK(christoffel.get(0, 0, 0) == approx(1.5));
  CHECK(christoffel.get(0, 0, 1) == approx(2.0));
  CHECK(christoffel.get(0, 1, 1) == approx(1.0));
  CHECK(christoffel.get(1, 0, 0) == approx(4.0));
  CHECK(christoffel.get(1, 0, 1) == approx(6.0));
  CHECK(christoffel.get(1, 1, 1) == approx(6.5));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::christoffel_first_kind(
          make_deriv_spatial_metric<spatial_dim>(used_for_size)),
      christoffel);
}

void test_3d_spatial_christoffel_first_kind(const DataVector &used_for_size) {
  const size_t spatial_dim = 3;
  const auto christoffel = gr::christoffel_first_kind(
      make_deriv_spatial_metric<spatial_dim>(0.));
  CHECK(christoffel.get(0, 0, 0) == approx(1.5));
  CHECK(christoffel.get(0, 0, 1) == approx(2.0));
  CHECK(christoffel.get(0, 0, 2) == approx(2.5));
  CHECK(christoffel.get(0, 1, 1) == approx(1.0));
  CHECK(christoffel.get(0, 1, 2) == approx(0.0));
  CHECK(christoffel.get(0, 2, 2) == approx(-2.5));
  CHECK(christoffel.get(1, 0, 0) == approx(4.0));
  CHECK(christoffel.get(1, 0, 1) == approx(6.0));
  CHECK(christoffel.get(1, 0, 2) == approx(8.0));
  CHECK(christoffel.get(1, 1, 1) == approx(6.5));
  CHECK(christoffel.get(1, 1, 2) == approx(7.0));
  CHECK(christoffel.get(1, 2, 2) == approx(6.0));
  CHECK(christoffel.get(2, 0, 0) == approx(6.5));
  CHECK(christoffel.get(2, 0, 1) == approx(10.0));
  CHECK(christoffel.get(2, 0, 2) == approx(13.5));
  CHECK(christoffel.get(2, 1, 1) == approx(12.0));
  CHECK(christoffel.get(2, 1, 2) == approx(14.0));
  CHECK(christoffel.get(2, 2, 2) == approx(14.5));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::christoffel_first_kind(
          make_deriv_spatial_metric<spatial_dim>(used_for_size)),
      christoffel);
}

void test_1d_spacetime_christoffel_first_kind(const DataVector &used_for_size) {
  const size_t spatial_dim = 1;
  const auto christoffel = gr::christoffel_first_kind(
      make_spacetime_deriv_spacetime_metric<spatial_dim>(0.));
  CHECK(christoffel.get(0, 0, 0) == approx(1.5));
  CHECK(christoffel.get(0, 0, 1) == approx(2.0));
  CHECK(christoffel.get(0, 1, 1) == approx(2.0));
  CHECK(christoffel.get(1, 0, 0) == approx(4.0));
  CHECK(christoffel.get(1, 0, 1) == approx(6.0));
  CHECK(christoffel.get(1, 1, 1) == approx(8.0));


  check_tensor_doubles_equals_tensor_datavectors(
      gr::christoffel_first_kind(
          make_spacetime_deriv_spacetime_metric<spatial_dim>(used_for_size)),
      christoffel);
}

void test_2d_spacetime_christoffel_first_kind(const DataVector &used_for_size) {
  const size_t spatial_dim = 2;
  const auto christoffel = gr::christoffel_first_kind(
      make_spacetime_deriv_spacetime_metric<spatial_dim>(0.));
  CHECK(christoffel.get(0, 0, 0) == approx(1.5));
  CHECK(christoffel.get(0, 0, 1) == approx(2.0));
  CHECK(christoffel.get(0, 0, 2) == approx(2.5));
  CHECK(christoffel.get(0, 1, 1) == approx(2.0));
  CHECK(christoffel.get(0, 1, 2) == approx(2.0));
  CHECK(christoffel.get(0, 2, 2) == approx(1.5));
  CHECK(christoffel.get(1, 0, 0) == approx(4.0));
  CHECK(christoffel.get(1, 0, 1) == approx(6.0));
  CHECK(christoffel.get(1, 0, 2) == approx(8.0));
  CHECK(christoffel.get(1, 1, 1) == approx(8.0));
  CHECK(christoffel.get(1, 1, 2) == approx(10.0));
  CHECK(christoffel.get(1, 2, 2) == approx(12.0));
  CHECK(christoffel.get(2, 0, 0) == approx(6.5));
  CHECK(christoffel.get(2, 0, 1) == approx(10.0));
  CHECK(christoffel.get(2, 0, 2) == approx(13.5));
  CHECK(christoffel.get(2, 1, 1) == approx(14.0));
  CHECK(christoffel.get(2, 1, 2) == approx(18.0));
  CHECK(christoffel.get(2, 2, 2) == approx(22.5));


  check_tensor_doubles_equals_tensor_datavectors(
      gr::christoffel_first_kind(
          make_spacetime_deriv_spacetime_metric<spatial_dim>(used_for_size)),
      christoffel);
}

void test_3d_spacetime_christoffel_first_kind(const DataVector &used_for_size) {
  const size_t spatial_dim = 3;
  const auto christoffel = gr::christoffel_first_kind(
      make_spacetime_deriv_spacetime_metric<spatial_dim>(0.));
  CHECK(christoffel.get(0, 0, 0) == approx(1.5));
  CHECK(christoffel.get(0, 0, 1) == approx(2.));
  CHECK(christoffel.get(0, 0, 2) == approx(2.5));
  CHECK(christoffel.get(0, 0, 3) == approx(3.));
  CHECK(christoffel.get(0, 1, 1) == approx(2.));
  CHECK(christoffel.get(0, 1, 2) == approx(2.));
  CHECK(christoffel.get(0, 1, 3) == approx(2.));
  CHECK(christoffel.get(0, 2, 2) == approx(1.5));
  CHECK(christoffel.get(0, 2, 3) == approx(1.));
  CHECK(christoffel.get(0, 3, 3) == approx(0.));
  CHECK(christoffel.get(1, 0, 0) == approx(4.));
  CHECK(christoffel.get(1, 0, 1) == approx(6.));
  CHECK(christoffel.get(1, 0, 2) == approx(8.));
  CHECK(christoffel.get(1, 0, 3) == approx(10.));
  CHECK(christoffel.get(1, 1, 1) == approx(8.));
  CHECK(christoffel.get(1, 1, 2) == approx(10.));
  CHECK(christoffel.get(1, 1, 3) == approx(12.));
  CHECK(christoffel.get(1, 2, 2) == approx(12.));
  CHECK(christoffel.get(1, 2, 3) == approx(14.));
  CHECK(christoffel.get(1, 3, 3) == approx(16.));
  CHECK(christoffel.get(2, 0, 0) == approx(6.5));
  CHECK(christoffel.get(2, 0, 1) == approx(10.));
  CHECK(christoffel.get(2, 0, 2) == approx(13.5));
  CHECK(christoffel.get(2, 0, 3) == approx(17.));
  CHECK(christoffel.get(2, 1, 1) == approx(14.));
  CHECK(christoffel.get(2, 1, 2) == approx(18.));
  CHECK(christoffel.get(2, 1, 3) == approx(22.));
  CHECK(christoffel.get(2, 2, 2) == approx(22.5));
  CHECK(christoffel.get(2, 2, 3) == approx(27.));
  CHECK(christoffel.get(2, 3, 3) == approx(32.));
  CHECK(christoffel.get(3, 0, 0) == approx(9.));
  CHECK(christoffel.get(3, 0, 1) == approx(14.));
  CHECK(christoffel.get(3, 0, 2) == approx(19.));
  CHECK(christoffel.get(3, 0, 3) == approx(24.));
  CHECK(christoffel.get(3, 1, 1) == approx(20.));
  CHECK(christoffel.get(3, 1, 2) == approx(26.));
  CHECK(christoffel.get(3, 1, 3) == approx(32.));
  CHECK(christoffel.get(3, 2, 2) == approx(33.));
  CHECK(christoffel.get(3, 2, 3) == approx(40.));
  CHECK(christoffel.get(3, 3, 3) == approx(48.));

  check_tensor_doubles_equals_tensor_datavectors(
      gr::christoffel_first_kind(
          make_spacetime_deriv_spacetime_metric<spatial_dim>(used_for_size)),
      christoffel);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Christoffel.",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv(2);
  test_1d_spatial_christoffel_first_kind(dv);
  test_2d_spatial_christoffel_first_kind(dv);
  test_3d_spatial_christoffel_first_kind(dv);
  test_1d_spacetime_christoffel_first_kind(dv);
  test_2d_spacetime_christoffel_first_kind(dv);
  test_3d_spacetime_christoffel_first_kind(dv);
}
