// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Triggers/Times.hpp"

namespace Triggers {
#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID Times::my_PUP_ID = 0;  // NOLINT
#endif                                   // SPECTRE_USE_CHARM
}  // namespace Triggers
