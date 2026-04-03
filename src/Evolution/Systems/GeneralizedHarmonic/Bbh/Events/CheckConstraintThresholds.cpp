// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Events/CheckConstraintThresholds.hpp"

namespace gh::bbh::Events {
#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID CheckConstraintThresholds::my_PUP_ID = 0;  // NOLINT
#endif  // SPECTRE_USE_CHARM
}  // namespace gh::bbh::Events
