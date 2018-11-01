// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"

#include <algorithm>
#include <pup.h>

#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace intrp {
namespace OptionHolders {
KerrHorizon::KerrHorizon(size_t l_max_in, std::array<double, 3> center_in,
                         double mass_in,
                         std::array<double, 3> dimensionless_spin_in,
                         const OptionContext& context)
    : l_max(l_max_in),
      center(std::move(center_in)),  // NOLINT
      mass(mass_in),
      dimensionless_spin(std::move(dimensionless_spin_in)) {  // NOLINT
  // above NOLINTs for std::move of trivially copyable type.
  if (mass <= 0.0) {
    // Check here, rather than put a lower_bound on the Tag, because
    // we want to exclude mass being exactly zero.
    PARSE_ERROR(context, "KerrHorizon expects mass>0, not " << mass);
  }
  if (magnitude(dimensionless_spin) > 1.0) {
    PARSE_ERROR(context, "KerrHorizon expects |dimensionless_spin|<=1, not "
                             << magnitude(dimensionless_spin));
  }
}

void KerrHorizon::pup(PUP::er& p) noexcept {
  p | l_max;
  p | center;
  p | mass;
  p | dimensionless_spin;
}

bool operator==(const KerrHorizon& lhs, const KerrHorizon& rhs) noexcept {
  return lhs.l_max == rhs.l_max and lhs.center == rhs.center and
         lhs.mass == rhs.mass and
         lhs.dimensionless_spin == rhs.dimensionless_spin;
}

bool operator!=(const KerrHorizon& lhs, const KerrHorizon& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace OptionHolders
}  // namespace intrp
