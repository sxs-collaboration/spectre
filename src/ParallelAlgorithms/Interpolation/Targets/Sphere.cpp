// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"

#include <array>
#include <cstddef>
#include <pup.h>

#include "Options/Options.hpp"

namespace intrp::OptionHolders {
Sphere::Sphere(const size_t l_max_in, const std::array<double, 3> center_in,
               const double radius_in)
    : l_max(l_max_in), center(center_in), radius(radius_in) {}

void Sphere::pup(PUP::er& p) {
  p | l_max;
  p | center;
  p | radius;
}

bool operator==(const Sphere& lhs, const Sphere& rhs) {
  return lhs.l_max == rhs.l_max and lhs.center == rhs.center and
         lhs.radius == rhs.radius;
}

bool operator!=(const Sphere& lhs, const Sphere& rhs) {
  return not(lhs == rhs);
}

}  // namespace intrp::OptionHolders
