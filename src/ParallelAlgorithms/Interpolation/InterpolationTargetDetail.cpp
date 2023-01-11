// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"

#include "DataStructures/LinkedMessageId.hpp"
#include "Time/TimeStepId.hpp"

namespace intrp::InterpolationTarget_detail {
double get_temporal_id_value(const double time) { return time; }
double get_temporal_id_value(const LinkedMessageId<double>& id) {
  return id.id;
}
double get_temporal_id_value(const TimeStepId& time_id) {
  return time_id.substep_time();
}
}  // namespace intrp::InterpolationTarget_detail
