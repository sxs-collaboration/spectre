// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/TestTrigger.hpp"

namespace TestHelpers::DenseTriggers {
#ifndef __CUDA_ARCH__
PUP::able::PUP_ID TestTrigger::my_PUP_ID = 0;  // NOLINT
#endif                                         // __CUDA_ARCH__
}  // namespace TestHelpers::DenseTriggers
