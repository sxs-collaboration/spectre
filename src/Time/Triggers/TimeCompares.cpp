// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Triggers/TimeCompares.hpp"

namespace Triggers {
#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID TimeCompares::my_PUP_ID = 0;  // NOLINT
#endif                                          // SPECTRE_USE_CHARM
}  // namespace Triggers
