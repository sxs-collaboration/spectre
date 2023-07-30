// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/Mass.hpp"

#include <cmath>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace StrahlkorperGr {
double irreducible_mass(const double area) {
  ASSERT(area > 0.0,
         "The area of the horizon must be greater than zero but is " << area);
  return sqrt(area / (16.0 * M_PI));
}

double christodoulou_mass(const double dimensionful_spin_magnitude,
                          const double irreducible_mass) {
  return sqrt(square(irreducible_mass) + (square(dimensionful_spin_magnitude) /
                                          (4.0 * square(irreducible_mass))));
}
}  // namespace StrahlkorperGr
