// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "StrahlkorperGrTestHelpers.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"

Strahlkorper<Frame::Inertial> create_strahlkorper_y11(
    const double y11_amplitude, const double radius,
    const std::array<double, 3>& center) {
  static const size_t l_max = 4;
  static const size_t m_max = 4;

  Strahlkorper<Frame::Inertial> strahlkorper_sphere(l_max, m_max, radius,
                                                    center);

  auto coefs = strahlkorper_sphere.coefficients();
  SpherepackIterator it(l_max, m_max);
  // Conversion between SPHEREPACK b_lm and real valued harmonic coefficients:
  // b_lm = (-1)^{m+1} sqrt(1/2pi) d_lm
  coefs[it.set(1, -1)()] = y11_amplitude * sqrt(0.5 / M_PI);
  return Strahlkorper<Frame::Inertial>(coefs, strahlkorper_sphere);
}
