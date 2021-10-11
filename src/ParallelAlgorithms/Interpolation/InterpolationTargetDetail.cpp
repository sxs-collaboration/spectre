// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"

namespace intrp::InterpolationTarget_detail {
double get_temporal_id_value(const double time) { return time; }
double evaluate_temporal_id_for_expiration(const double time) { return time; }
}  // namespace intrp::InterpolationTarget_detail
