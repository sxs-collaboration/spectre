// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/RadiusFunctions.hpp"

#include <cmath>
#include <cstddef>

#include "Utilities/ErrorHandling/Error.hpp"

namespace CurvedScalarWave::Worldtube {

namespace detail {

void check_alpha(const double alpha) {
  if (alpha < 0. or alpha > 4.) {
    ERROR(
        "Only exponents between 0 and 4 have been tested for the broken power "
        "law. Values outside this range are likely not sensible.");
  }
}

void check_delta(const double delta) {
  if (delta < 1e-2) {
    ERROR(
        "A value of delta less than 0.01 is disabled as it can lead to "
        "prohibitively large floating precision errors.");
  }
}
}  // namespace detail

double smooth_broken_power_law(const double orbit_radius, const double alpha,
                               const double amp, const double rb,
                               const double delta) {
  detail::check_alpha(alpha);
  detail::check_delta(delta);
  const double r_by_rb = orbit_radius / rb;
  return amp * pow(r_by_rb, alpha) *
         pow(0.5 * (1. + pow(r_by_rb, 1. / delta)), -alpha * delta);
}

double smooth_broken_power_law_derivative(const double orbit_radius,
                                          const double alpha, const double amp,
                                          const double rb, const double delta) {
  detail::check_alpha(alpha);
  detail::check_delta(delta);
  // During testing it was found that for large values of rb, the equation below
  // can FPE even though the derivative is clearly zero up to double precision.
  // To avoid this, we introduce this shortcut.
  if (orbit_radius > rb + 1000. * delta) {
    return 0.;
  }
  const double r_by_rb = orbit_radius / rb;
  const double temp1 = pow(r_by_rb, 1. / delta - 1.);
  const double temp2 = 0.5 * temp1 * r_by_rb + 0.5;

  return amp * alpha * pow(r_by_rb, alpha - 1) / rb *
         pow(temp2, -alpha * delta - 1.) * (temp2 - 0.5 * temp1 * r_by_rb);
}

}  // namespace CurvedScalarWave::Worldtube
