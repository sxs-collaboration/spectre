// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/IncreaseResolution.hpp"

namespace ah::Criteria {
bool IncreaseResolution::is_equal(const Criterion& other) const {
  return dynamic_cast<const IncreaseResolution*>(&other) != nullptr;
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID IncreaseResolution::my_PUP_ID = 0;  // NOLINT
#endif                                                // SPECTRE_USE_CHARM
}  // namespace ah::Criteria
