// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Math.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Spectral {
namespace Swsh {
namespace TestHelpers {
SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshTestHelpers",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> theta_dist{-M_PI / 2.0, M_PI / 2.0};
  UniformCustomDistribution<double> phi_dist{0, 2.0 * M_PI};
  UniformCustomDistribution<size_t> factorial_dist{0, 20};

  size_t test_factorial_value = factorial_dist(gen);
  CHECK(approx(factorial(test_factorial_value)) ==
        static_cast<double>(::factorial(test_factorial_value)));

  const double theta_point = theta_dist(gen);
  const double phi_point = phi_dist(gen);
  const std::complex<double> expected_swsh_02m2 =
      sqrt(15.0 / (32.0 * M_PI)) * sin(theta_point) * sin(theta_point) *
      (std::complex<double>(cos(-2.0 * phi_point), sin(-2.0 * phi_point)));

  const std::complex<double> expected_swsh_110 =
      sqrt(3.0 / (8.0 * M_PI)) * sin(theta_point);

  const std::complex<double> expected_swsh_111 =
      -sqrt(3.0 / (16.0 * M_PI)) * (1 - cos(theta_point)) *
      (std::complex<double>(cos(phi_point), sin(phi_point)));

  CHECK(real(spin_weighted_spherical_harmonic(0, 2, -2, theta_point,
                                              phi_point)) ==
        approx(real(expected_swsh_02m2)));
  CHECK(imag(spin_weighted_spherical_harmonic(0, 2, -2, theta_point,
                                              phi_point)) ==
        approx(imag(expected_swsh_02m2)));

  CHECK(
      real(spin_weighted_spherical_harmonic(1, 1, 0, theta_point, phi_point)) ==
      approx(real(expected_swsh_110)));
  CHECK(
      imag(spin_weighted_spherical_harmonic(1, 1, 0, theta_point, phi_point)) ==
      approx(imag(expected_swsh_110)));

  CHECK(
      real(spin_weighted_spherical_harmonic(1, 1, 1, theta_point, phi_point)) ==
      approx(real(expected_swsh_111)));
  CHECK(
      imag(spin_weighted_spherical_harmonic(1, 1, 1, theta_point, phi_point)) ==
      approx(imag(expected_swsh_111)));
}
}  // namespace TestHelpers
}  // namespace Swsh
}  // namespace Spectral
