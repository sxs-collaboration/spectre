// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/range/combine.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_compute_spacetime_metric(const DataVector& structure) {
  const auto psi = compute_spacetime_metric(make_lapse(0.), make_shift(0.),
                                            make_spatial_metric(0.));
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
      compute_spacetime_metric(make_lapse(structure),
                               make_shift(structure),
                               make_spatial_metric(structure)),
      psi);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrFunctions.SpacetimeDecomp",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv(2);
  test_compute_spacetime_metric(dv);
}
