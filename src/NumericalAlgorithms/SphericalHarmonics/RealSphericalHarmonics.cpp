// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/RealSphericalHarmonics.hpp"

#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

void real_spherical_harmonic(gsl::not_null<DataVector*> spherical_harmonic,
                             const DataVector& theta, const DataVector& phi,
                             size_t l, int m) {
  ASSERT(size_t(abs(m)) <= l, "m needs to be smaller than l");
  const int sign = (m % 2 == 0) ? 1 : -1;
  const double prefactor = sign * M_SQRT2;
  if (m < 0) {
    for (size_t i = 0; i < theta.size(); ++i) {
      (*spherical_harmonic)[i] = prefactor * boost::math::spherical_harmonic_i(
                                                 l, abs(m), theta[i], phi[i]);
    }
  } else if (m > 0) {
    for (size_t i = 0; i < theta.size(); ++i) {
      (*spherical_harmonic)[i] =
          prefactor * boost::math::spherical_harmonic_r(l, m, theta[i], phi[i]);
    }
  } else {
    for (size_t i = 0; i < theta.size(); ++i) {
      (*spherical_harmonic)[i] =
          boost::math::spherical_harmonic_r(l, m, theta[i], phi[i]);
    }
  }
}

DataVector real_spherical_harmonic(const DataVector& theta,
                                   const DataVector& phi, size_t l, int m) {
  DataVector result(theta.size());
  real_spherical_harmonic(make_not_null(&result), theta, phi, l, m);
  return result;
}
