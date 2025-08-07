// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/IncreaseResolution.hpp"

namespace ah::Criteria {
IncreaseResolution::IncreaseResolution(CkMigrateMessage* msg)
    : Criterion(msg) {}

bool IncreaseResolution::is_equal(const Criterion& other) const {
  return dynamic_cast<const IncreaseResolution*>(&other) != nullptr;
}

PUP::able::PUP_ID IncreaseResolution::my_PUP_ID = 0;  // NOLINT
}  // namespace ah::Criteria
