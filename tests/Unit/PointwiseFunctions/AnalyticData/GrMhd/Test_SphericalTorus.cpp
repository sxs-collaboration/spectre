// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SphericalTorus.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
std::array<double, 3> t2a(const tnsr::I<double, 3>& tens) {
  std::array<double, 3> arr{};
  for (size_t i = 0; i < 3; i++) {
    gsl::at(arr,i) = tens.get(i);
  }
  return arr;
}

tnsr::I<double, 3> a2t(const std::array<double, 3>& arr) {
  tnsr::I<double, 3> tens;
  for (size_t i = 0; i < 3; i++) {
    tens.get(i) = gsl::at(arr,i);
  }
  return tens;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.SphericalTorus",
                  "[Unit][PointwiseFunctions]") {
  // check parse errors first
  CHECK_THROWS_WITH(
      ([]() { grmhd::AnalyticData::SphericalTorus(-1.0, 2.0, 0.1, 0.5); })(),
      Catch::Matchers::ContainsSubstring("Minimum radius must be positive."));
  CHECK_THROWS_WITH(
      ([]() { grmhd::AnalyticData::SphericalTorus(3.0, 2.0, 0.1, 0.5); })(),
      Catch::Matchers::ContainsSubstring(
          "Maximum radius must be greater than minimum radius."));
  CHECK_THROWS_WITH(
      ([]() { grmhd::AnalyticData::SphericalTorus(1.0, 2.0, 10.0, 0.5); })(),
      Catch::Matchers::ContainsSubstring(
          "Minimum polar angle should be less than pi/2."));
  CHECK_THROWS_WITH(
      ([]() { grmhd::AnalyticData::SphericalTorus(1.0, 2.0, -0.1, 0.5); })(),
      Catch::Matchers::ContainsSubstring(
          "Minimum polar angle should be positive"));
  CHECK_THROWS_WITH(
      ([]() { grmhd::AnalyticData::SphericalTorus(1.0, 2.0, 0.1, -0.5); })(),
      Catch::Matchers::ContainsSubstring(
          "Fraction of torus included must be positive."));
  CHECK_THROWS_WITH(
      ([]() { grmhd::AnalyticData::SphericalTorus(1.0, 2.0, 0.1, 2.0); })(),
      Catch::Matchers::ContainsSubstring(
          "Fraction of torus included must be at most 1."));

  // check constructor
  CHECK(grmhd::AnalyticData::SphericalTorus(std::array<double, 2>{1.0, 2.0},
                                            0.1, 0.5) ==
        grmhd::AnalyticData::SphericalTorus(1.0, 2.0, 0.1, 0.5));

  MAKE_GENERATOR(gen);
  const double r_min = std::uniform_real_distribution<>(1.0, 2.0)(gen);
  const double r_max = std::uniform_real_distribution<>(3.0, 4.0)(gen);
  const double phi_max = std::uniform_real_distribution<>(0.5, 1.0)(gen);
  const double fraction_of_torus =
      std::uniform_real_distribution<>(0.1, 1.0)(gen);

  const grmhd::AnalyticData::SphericalTorus full_torus(r_min, r_max, phi_max);
  const grmhd::AnalyticData::SphericalTorus partial_torus(r_min, r_max, phi_max,
                                                          fraction_of_torus);
  {
    const double x = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double y = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const tnsr::I<double, 3> test_pt1{{{x, y, -1.0}}};
    const tnsr::I<double, 3> test_pt2{{{x, y, 1.0}}};
    CHECK_ITERABLE_APPROX(full_torus(test_pt1), full_torus(test_pt2));
  }

  {
    const double x = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double y1 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double y2 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double z1 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double z2 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);

    tnsr::I<double, 3> test_coords1{{{x, y1, z1}}};
    tnsr::I<double, 3> test_coords2{{{x, y2, z2}}};
    test_coords1 = partial_torus(test_coords1);
    test_coords2 = partial_torus(test_coords2);

    CHECK(magnitude(t2a(test_coords1)) == approx(magnitude(t2a(test_coords2))));
  }

  CHECK(full_torus == full_torus);
  CHECK_FALSE(full_torus != full_torus);
  CHECK(partial_torus == partial_torus);
  CHECK_FALSE(partial_torus != partial_torus);
  CHECK_FALSE(full_torus == partial_torus);
  CHECK(full_torus != partial_torus);

  CHECK(full_torus !=
        grmhd::AnalyticData::SphericalTorus(r_min + 0.1, r_max, phi_max));
  CHECK(full_torus !=
        grmhd::AnalyticData::SphericalTorus(r_min, r_max + 0.1, phi_max));
  CHECK(full_torus !=
        grmhd::AnalyticData::SphericalTorus(r_min, r_max, phi_max + 0.1));

  CHECK(not full_torus.is_identity());
  CHECK(not partial_torus.is_identity());

  {
    auto deriv_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    const auto test_point = make_with_random_values<std::array<double, 3>>(
        make_not_null(&gen), make_not_null(&dist), double{});
    const auto test_tnsr = a2t(test_point);
    const tnsr::I<double, 3> mapped_tnsr = partial_torus(test_tnsr);
    const auto mapped_point = t2a(mapped_tnsr);

    const tnsr::I<double, 3> iden_tnsr = partial_torus.inverse(mapped_tnsr);

    const auto jac = partial_torus.jacobian(test_tnsr);
    const auto jac_inv = partial_torus.inv_jacobian(test_tnsr);
    const auto analytic = partial_torus.hessian(test_tnsr);
    const auto analytic_inverse =
        partial_torus.derivative_of_inv_jacobian(test_tnsr);
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      CHECK(test_tnsr.get(i) == deriv_approx(iden_tnsr.get(i)));
      for (size_t j = 0; j < 3; ++j) {
        CAPTURE(j);
        const auto numerical_jacobian = numerical_derivative(
            [&i, &partial_torus](const std::array<double, 3>& x) {
              const auto x_tnsr = a2t(x);
              return partial_torus(x_tnsr).get(i);
            },
            test_point, j, 1e-3);

        const auto numerical_inverse_jacobian = numerical_derivative(
            [&i, &partial_torus](const std::array<double, 3>& mapped_x) {
              const auto mapped_x_tnsr = a2t(mapped_x);
              return partial_torus.inverse(mapped_x_tnsr).get(i);
            },
            mapped_point, j, 1e-3);

        CHECK(jac.get(i, j) == deriv_approx(numerical_jacobian));
        CHECK(jac_inv.get(i, j) == deriv_approx(numerical_inverse_jacobian));
        for (size_t k = 0; k < 3; ++k) {
          CAPTURE(k);
          const auto numerical_hessian = numerical_derivative(
              [&i, &j, &partial_torus](const std::array<double, 3>& y) {
                const auto y_tnsr = a2t(y);
                return std::array{partial_torus.jacobian(y_tnsr).get(i, j),
                                  partial_torus.inv_jacobian(y_tnsr).get(i, j)};
              },
              test_point, k, 1e-3);
          CHECK(analytic.get(i, j, k) == deriv_approx(numerical_hessian[0]));
          CHECK(analytic_inverse.get(i, j, k) ==
                deriv_approx(numerical_hessian[1]));
        }
      }
    }
  }
}
