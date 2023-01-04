// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Punctures/Flatness.hpp"

namespace Punctures::Solutions {
bool operator==(const Flatness& /*lhs*/, const Flatness& /*rhs*/) {
  return true;
}

bool operator!=(const Flatness& lhs, const Flatness& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID Flatness::my_PUP_ID = 0;  // NOLINT
}  // namespace Punctures::Solutions
