// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/Completion.hpp"

#if defined(SPECTRE_USE_CHARM)
namespace Events {
PUP::able::PUP_ID Completion::my_PUP_ID = 0;  // NOLINT
}  // namespace Events
#endif  // SPECTRE_USE_CHARM
