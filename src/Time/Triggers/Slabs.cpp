// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Triggers/Slabs.hpp"

namespace Triggers {
#ifndef __CUDA_ARCH__
PUP::able::PUP_ID Slabs::my_PUP_ID = 0;  // NOLINT
#endif                                   // __CUDA_ARCH__
}  // namespace Triggers
