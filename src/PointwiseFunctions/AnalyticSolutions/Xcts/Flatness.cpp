// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"

namespace Xcts::Solutions {
bool operator==(const Flatness& /*lhs*/, const Flatness& /*rhs*/) {
  return true;
}

bool operator!=(const Flatness& lhs, const Flatness& rhs) {
  return not(lhs == rhs);
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID Flatness::my_PUP_ID = 0;  // NOLINT
#endif                                      // SPECTRE_USE_CHARM
}  // namespace Xcts::Solutions
