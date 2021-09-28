// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Triggers/NearTimes.hpp"

namespace Triggers {
bool NearTimes::operator()(const double now, const TimeDelta& time_step) const {
  const bool time_runs_forward = time_step.is_positive();

  double range_code_units = range_;
  if (unit_ == Unit::Slab) {
    range_code_units *= time_step.slab().duration().value();
  } else if (unit_ == Unit::Step) {
    range_code_units *= std::abs(time_step.value());
  }

  if (not time_runs_forward) {
    range_code_units = -range_code_units;
  }

  // Interval around now to look for trigger times in.
  auto trigger_range = std::make_pair(
      direction_ == Direction::Before ? now : now - range_code_units,
      direction_ == Direction::After ? now : now + range_code_units);

  if (not time_runs_forward) {
    std::swap(trigger_range.first, trigger_range.second);
  }

  const auto nearby_times = times_->times_near(trigger_range.first);
  for (const auto& time : nearby_times) {
    if (time and *time >= trigger_range.first and
        *time <= trigger_range.second) {
      return true;
    }
  }
  return false;
}

void NearTimes::pup(PUP::er& p) {
  p | times_;
  p | range_;
  p | unit_;
  p | direction_;
}

PUP::able::PUP_ID NearTimes::my_PUP_ID = 0;  // NOLINT
}  // namespace Triggers
