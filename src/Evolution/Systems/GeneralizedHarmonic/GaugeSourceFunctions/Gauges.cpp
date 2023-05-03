// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"

namespace gh::gauges {
GaugeCondition::GaugeCondition(CkMigrateMessage* msg) : PUP::able(msg) {}
}  // namespace gh::gauges
