// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/TimeStepId.hpp"

namespace SelfStart {
/// Reports whether the `time_id` is during self start
///
/// This currently assumes that the slab number of the `time_id` will be
/// negative if and only if self-start is in progress. If self start is
/// modified to alter that behavior, this utility must also be modified.
bool is_self_starting(const TimeStepId& time_id) noexcept;
}  // namespace SelfStart
