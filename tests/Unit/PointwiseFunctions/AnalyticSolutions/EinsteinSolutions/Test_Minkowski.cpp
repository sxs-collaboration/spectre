// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/Minkowski.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim, typename T>
void test_minkowski(const T& value) {
  EinsteinSolutions::Minkowski<Dim> minkowski{};

  const tnsr::I<T, Dim> x{value};
  const double t = 1.2;

  const auto one = make_with_value<T>(value, 1.);
  const auto zero = make_with_value<T>(value, 0.);

  const auto lapse = minkowski.lapse(x, t);
  const auto dt_lapse = minkowski.dt_lapse(x, t);
  const auto d_lapse = minkowski.deriv_lapse(x, t);
  const auto shift = minkowski.shift(x, t);
  const auto deriv_shift = minkowski.deriv_shift(x, t);
  const auto g = minkowski.spatial_metric(x, t);
  const auto dt_g = minkowski.dt_spatial_metric(x, t);
  const auto d_g = minkowski.deriv_spatial_metric(x, t);
  const auto det_g = minkowski.sqrt_determinant_of_spatial_metric(x, t);
  const auto dt_det_g = minkowski.dt_sqrt_determinant_of_spatial_metric(x, t);
  const auto inv_g = minkowski.inverse_spatial_metric(x, t);
  const auto extrinsic_curvature = minkowski.extrinsic_curvature(x, t);

  CHECK(lapse.get() == one);
  CHECK(dt_lapse.get() == zero);
  CHECK(det_g.get() == one);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(shift.get(i) == zero);
    CHECK(d_lapse.get(i) == zero);
    CHECK(g.get(i, i) == one);
    CHECK(inv_g.get(i, i) == one);
    for (size_t j = 0; j < i; ++j) {
      CHECK(g.get(i, j) == zero);
      CHECK(inv_g.get(i, j) == zero);
    }
    for (size_t j = 0; j < Dim; ++j) {
      CHECK(dt_g.get(i, j) == zero);
      CHECK(extrinsic_curvature.get(i, j) == zero);
      for (size_t k = 0; k < Dim; ++k) {
        CHECK(d_g.get(i, j, k) == zero);
      }
    }
  }
  test_serialization(minkowski);
  // test operator !=
  CHECK_FALSE(minkowski != minkowski);
}

template <size_t Dim>
void test_option_creation() {
  test_creation<EinsteinSolutions::Minkowski<Dim>>("");
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.EinsteinSolution.Minkowski",
    "[PointwiseFunctions][Unit]") {
  const double x = 1.2;
  const DataVector x_dv{1., 2., 3.};

  test_minkowski<1>(x);
  test_minkowski<1>(x_dv);
  test_minkowski<2>(x);
  test_minkowski<2>(x_dv);
  test_minkowski<3>(x);
  test_minkowski<3>(x_dv);

  test_option_creation<1>();
  test_option_creation<2>();
  test_option_creation<3>();
}
