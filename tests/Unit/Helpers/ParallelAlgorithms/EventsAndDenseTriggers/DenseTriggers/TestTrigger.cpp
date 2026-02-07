// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/TestTrigger.hpp"

#if defined(SPECTRE_USE_CHARM)
namespace TestHelpers::DenseTriggers {
PUP::able::PUP_ID TestTrigger::my_PUP_ID = 0;  // NOLINT
}  // namespace TestHelpers::DenseTriggers
#endif  // SPECTRE_USE_CHARM
