// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveAdaptiveSteppingDiagnostics.hpp"

namespace Events {
#ifndef __CUDA_ARCH__
PUP::able::PUP_ID ObserveAdaptiveSteppingDiagnostics::my_PUP_ID = 0;  // NOLINT
#endif  // __CUDA_ARCH__
}  // namespace Events
