// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>

#include "Domain/CoordinateMaps/SphericalTorus.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/StdArrayHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SphericalTorus",
                  "[Domain][Unit]") {
  // check parse errors first
  CHECK_THROWS_WITH(
      ([]() { domain::CoordinateMaps::SphericalTorus(-1.0, 2.0, 0.1, 0.5); })(),
      Catch::Contains("Minimum radius must be positive."));
  CHECK_THROWS_WITH(
      ([]() { domain::CoordinateMaps::SphericalTorus(3.0, 2.0, 0.1, 0.5); })(),
      Catch::Contains("Maximum radius must be greater than minimum radius."));
  CHECK_THROWS_WITH(
      ([]() { domain::CoordinateMaps::SphericalTorus(1.0, 2.0, 10.0, 0.5); })(),
      Catch::Contains("Minimum polar angle should be less than pi/2."));
  CHECK_THROWS_WITH(
      ([]() { domain::CoordinateMaps::SphericalTorus(1.0, 2.0, -0.1, 0.5); })(),
      Catch::Contains("Minimum polar angle should be positive"));
  CHECK_THROWS_WITH(
      ([]() { domain::CoordinateMaps::SphericalTorus(1.0, 2.0, 0.1, -0.5); })(),
      Catch::Contains("Fraction of torus included must be positive."));
  CHECK_THROWS_WITH(
      ([]() { domain::CoordinateMaps::SphericalTorus(1.0, 2.0, 0.1, 2.0); })(),
      Catch::Contains("Fraction of torus included must be at most 1."));

  // check constructor
  CHECK(domain::CoordinateMaps::SphericalTorus(std::array<double, 2>{1.0, 2.0},
                                               0.1, 0.5) ==
        domain::CoordinateMaps::SphericalTorus(1.0, 2.0, 0.1, 0.5));

  MAKE_GENERATOR(gen);
  const double r_min = std::uniform_real_distribution<>(1.0, 2.0)(gen);
  const double r_max = std::uniform_real_distribution<>(3.0, 4.0)(gen);
  const double phi_max = std::uniform_real_distribution<>(0.5, 1.0)(gen);
  const double fraction_of_torus =
      std::uniform_real_distribution<>(0.1, 1.0)(gen);

  const domain::CoordinateMaps::SphericalTorus full_torus(r_min, r_max,
                                                          phi_max);
  const domain::CoordinateMaps::SphericalTorus partial_torus(
      r_min, r_max, phi_max, fraction_of_torus);

  // Can't do for full_torus because it is not invertable on the boundary.
  test_suite_for_map_on_unit_cube(partial_torus);

  {
    const double x = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double y = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    CHECK_ITERABLE_APPROX(full_torus(std::array{x, y, -1.0}),
                          full_torus(std::array{x, y, 1.0}));
  }
  {
    const double x = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double y1 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double y2 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double z1 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    const double z2 = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
    CHECK(magnitude(partial_torus(std::array{x, y1, z1})) ==
          approx(magnitude(partial_torus(std::array{x, y2, z2}))));
  }

  test_coordinate_map_implementation(full_torus);
  test_coordinate_map_implementation(partial_torus);

  CHECK(full_torus == full_torus);
  CHECK_FALSE(full_torus != full_torus);
  CHECK(partial_torus == partial_torus);
  CHECK_FALSE(partial_torus != partial_torus);
  CHECK_FALSE(full_torus == partial_torus);
  CHECK(full_torus != partial_torus);

  CHECK(full_torus !=
        domain::CoordinateMaps::SphericalTorus(r_min + 0.1, r_max, phi_max));
  CHECK(full_torus !=
        domain::CoordinateMaps::SphericalTorus(r_min, r_max + 0.1, phi_max));
  CHECK(full_torus !=
        domain::CoordinateMaps::SphericalTorus(r_min, r_max, phi_max + 0.1));

  CHECK(not full_torus.is_identity());
  CHECK(not partial_torus.is_identity());

  {
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    const auto test_point = make_with_random_values<std::array<double, 3>>(
        make_not_null(&gen), make_not_null(&dist), double{});

    const auto analytic = partial_torus.hessian(test_point);
    const auto analytic_inverse =
        partial_torus.derivative_of_inv_jacobian(test_point);
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      for (size_t j = 0; j < 3; ++j) {
        CAPTURE(j);
        for (size_t k = 0; k < 3; ++k) {
          CAPTURE(k);
          const auto numerical = numerical_derivative(
              [&i, &j, &partial_torus](const std::array<double, 3>& x) {
                return std::array{partial_torus.jacobian(x).get(i, j),
                                  partial_torus.inv_jacobian(x).get(i, j)};
              },
              test_point, k, 1e-2);
          auto deriv_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);
          CHECK(analytic.get(i, j, k) == deriv_approx(numerical[0]));
          CHECK(analytic_inverse.get(i, j, k) == deriv_approx(numerical[1]));
        }
      }
    }
  }
}
