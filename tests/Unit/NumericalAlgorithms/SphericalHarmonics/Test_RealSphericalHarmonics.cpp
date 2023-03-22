// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/RealSphericalHarmonics.hpp"

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.RealSphericalHarmonics",
                  "[NumericalAlgorithms][Unit]") {
  const size_t l_max = 10;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> phi_distribution(0., 2. * M_PI);
  std::uniform_real_distribution<> theta_distribution(0., M_PI);
  const size_t num_points = 100;
  const auto thetas = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&theta_distribution),
      DataVector(num_points));
  const auto phis = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&phi_distribution),
      DataVector(num_points));
  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);

  for (size_t l = 0; l <= l_max; ++l) {
    for (int m = -l; m <= static_cast<int>(l); ++m) {
      const auto spherical_harmonic =
          real_spherical_harmonic(thetas, phis, l, m);
      if (m == 0) {
        const Spectral::Swsh::SpinWeightedSphericalHarmonic
            complex_spherical_harmonic(0, l, m);
        CHECK_ITERABLE_CUSTOM_APPROX(
            real(complex_spherical_harmonic.evaluate(
                thetas, phis, sin(thetas / 2.), cos(thetas / 2.))),
            spherical_harmonic, custom_approx);
      } else if (m > 0) {
        const Spectral::Swsh::SpinWeightedSphericalHarmonic
            complex_spherical_harmonic(0, l, m);
        const int sign = (m % 2 == 0) ? 1 : -1;
        CHECK_ITERABLE_CUSTOM_APPROX(
            real(complex_spherical_harmonic.evaluate(
                thetas, phis, sin(thetas / 2.), cos(thetas / 2.))) *
                M_SQRT2 * sign,
            spherical_harmonic, custom_approx);
      } else {
        const Spectral::Swsh::SpinWeightedSphericalHarmonic
            complex_spherical_harmonic(0, l, abs(m));
        const int sign = (m % 2 == 0) ? 1 : -1;
        CHECK_ITERABLE_CUSTOM_APPROX(
            imag(complex_spherical_harmonic.evaluate(
                thetas, phis, sin(thetas / 2.), cos(thetas / 2.))) *
                M_SQRT2 * sign,
            spherical_harmonic, custom_approx);
      }
    }
  }
}
