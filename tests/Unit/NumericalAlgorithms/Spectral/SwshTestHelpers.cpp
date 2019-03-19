// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"

#include <boost/math/special_functions/binomial.hpp>
#include <cmath>
#include <complex>

#include "NumericalAlgorithms/Spectral/SwshTags.hpp"  // IWYU pragma: keep

namespace Spectral {
namespace Swsh {
namespace TestHelpers {

double factorial(const size_t arg) noexcept {
  if (arg <= 1) {
    return 1.0;
  }
  double factorial_result = 1.0;
  for (size_t product_term = 1; product_term <= arg; ++product_term) {
    factorial_result *= static_cast<double>(product_term);
  }
  return factorial_result;
}

std::complex<double> spin_weighted_spherical_harmonic(
    const int s, const int l, const int m, const double theta,
    const double phi) noexcept {
  if (l + s < 0 or l - s < 0) {
    return 0.0;
  }
  std::complex<double> swshval = 0.0;
  for (int r = 0; r <= l - s; ++r) {
    if (r + s - m >= 0 and l - r + m >= 0) {
      swshval +=
          boost::math::binomial_coefficient<double>(
              static_cast<unsigned>(l - s), static_cast<unsigned>(r)) *
          boost::math::binomial_coefficient<double>(
              static_cast<unsigned>(l + s), static_cast<unsigned>(r + s - m)) *
          ((l - r - s) % 2 == 0 ? 1.0 : -1.0) *
          pow(cos(theta / 2.0) / sin(theta / 2.0), 2.0 * r + s - m);
    }
  }
  swshval *= (m % 2 == 0 ? 1.0 : -1.0) *
             sqrt(factorial(static_cast<size_t>(l) + static_cast<size_t>(m)) *
                  factorial(static_cast<size_t>(l - m)) * (2.0 * l + 1) /
                  (4.0 * M_PI *
                   factorial(static_cast<size_t>(l) + static_cast<size_t>(s)) *
                   factorial(static_cast<size_t>(l - s)))) *
             (std::complex<double>(cos(m * phi), sin(m * phi))) *
             pow(sin(theta / 2.0), 2.0 * l);
  return swshval;
}

template <>
std::complex<double> derivative_of_spin_weighted_spherical_harmonic<Tags::Eth>(
    const int s, const int l, const int m, const double theta,
    const double phi) noexcept {
  return sqrt(static_cast<std::complex<double>>((l - s) * (l + s + 1))) *
         spin_weighted_spherical_harmonic(s + 1, l, m, theta, phi);
}

template <>
std::complex<double>
derivative_of_spin_weighted_spherical_harmonic<Tags::Ethbar>(
    const int s, const int l, const int m, const double theta,
    const double phi) noexcept {
  return -sqrt(static_cast<std::complex<double>>((l + s) * (l - s + 1))) *
         spin_weighted_spherical_harmonic(s - 1, l, m, theta, phi);
}

template <>
std::complex<double>
derivative_of_spin_weighted_spherical_harmonic<Tags::EthEth>(
    const int s, const int l, const int m, const double theta,
    const double phi) noexcept {
  return sqrt(static_cast<std::complex<double>>((l - s) * (l + s + 1))) *
         derivative_of_spin_weighted_spherical_harmonic<Tags::Eth>(s + 1, l, m,
                                                                   theta, phi);
}

template <>
std::complex<double>
derivative_of_spin_weighted_spherical_harmonic<Tags::EthbarEth>(
    const int s, const int l, const int m, const double theta,
    const double phi) noexcept {
  return -sqrt(static_cast<std::complex<double>>((l + s) * (l - s + 1))) *
         derivative_of_spin_weighted_spherical_harmonic<Tags::Eth>(s - 1, l, m,
                                                                   theta, phi);
}

template <>
std::complex<double>
derivative_of_spin_weighted_spherical_harmonic<Tags::EthEthbar>(
    const int s, const int l, const int m, const double theta,
    const double phi) noexcept {
  return sqrt(static_cast<std::complex<double>>((l - s) * (l + s + 1))) *
         derivative_of_spin_weighted_spherical_harmonic<Tags::Ethbar>(
             s + 1, l, m, theta, phi);
}

template <>
std::complex<double>
derivative_of_spin_weighted_spherical_harmonic<Tags::EthbarEthbar>(
    const int s, const int l, const int m, const double theta,
    const double phi) noexcept {
  return -sqrt(static_cast<std::complex<double>>((l + s) * (l - s + 1))) *
         derivative_of_spin_weighted_spherical_harmonic<Tags::Ethbar>(
             s - 1, l, m, theta, phi);
}

}  // namespace TestHelpers
}  // namespace Swsh
}  // namespace Spectral
