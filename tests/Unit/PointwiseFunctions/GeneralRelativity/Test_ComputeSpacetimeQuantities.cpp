// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/range/combine.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_compute_1d_spacetime_metric(const DataVector &used_for_size) {
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

void test_compute_2d_spacetime_metric(const DataVector &used_for_size) {
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

void test_compute_3d_spacetime_metric(const DataVector &used_for_size) {
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
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrFunctions.SpacetimeDecomp",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv(2);
  test_compute_1d_spacetime_metric(dv);
  test_compute_2d_spacetime_metric(dv);
  test_compute_3d_spacetime_metric(dv);
}
