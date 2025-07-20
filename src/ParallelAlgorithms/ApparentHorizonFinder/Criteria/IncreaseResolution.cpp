// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/IncreaseResolution.hpp"

namespace ah::Criteria {
IncreaseResolution::IncreaseResolution(CkMigrateMessage* msg)
    : Criterion(msg) {}

#ifndef __CUDA_ARCH__
PUP::able::PUP_ID IncreaseResolution::my_PUP_ID = 0;  // NOLINT
#endif                                                // __CUDA_ARCH__
}  // namespace ah::Criteria
