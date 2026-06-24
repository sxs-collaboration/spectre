// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"

#include <ostream>
#include <pup.h>

namespace evolution::dg {
void EqualRateRegionId::pup(PUP::er& p) {
  p | type;
  p | label;
}

std::ostream& operator<<(std::ostream& os, const EqualRateRegionId& id) {
  return os << "{" << id.type << "," << id.label << "}";
}
}  // namespace evolution::dg
