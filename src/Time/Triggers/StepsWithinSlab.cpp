// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Triggers/StepsWithinSlab.hpp"

namespace Triggers {
#ifndef __CUDA_ARCH__
PUP::able::PUP_ID StepsWithinSlab::my_PUP_ID = 0;  // NOLINT
#endif                                             // __CUDA_ARCH__
}  // namespace Triggers
