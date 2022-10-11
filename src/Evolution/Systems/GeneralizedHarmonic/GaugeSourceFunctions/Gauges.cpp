// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"

namespace GeneralizedHarmonic::gauges {
GaugeCondition::GaugeCondition(CkMigrateMessage* msg) : PUP::able(msg) {}
}  // namespace GeneralizedHarmonic::gauges
