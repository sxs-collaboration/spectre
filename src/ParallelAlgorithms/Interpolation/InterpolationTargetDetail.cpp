// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"

#include "Time/TimeStepId.hpp"

namespace intrp::InterpolationTarget_detail {
double get_temporal_id_value(const double time) { return time; }
double get_temporal_id_value(const TimeStepId& time_id) {
  return time_id.substep_time().value();
}
double evaluate_temporal_id_for_expiration(const double time) { return time; }
double evaluate_temporal_id_for_expiration(const TimeStepId& time_id) {
  return time_id.step_time().value();
}
}  // namespace intrp::InterpolationTarget_detail
