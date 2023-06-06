// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Utilities.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
/// Suggests step sizes to place steps at specific times.
///
/// The suggestion provided depends on the current time, so it should
/// be applied immediately, rather than delayed several slabs.
class StepToTimes : public StepChooser<StepChooserUse::Slab> {
 public:
  /// \cond
  StepToTimes() = default;
  explicit StepToTimes(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(StepToTimes);  // NOLINT
  /// \endcond

  struct Times {
    using type = std::unique_ptr<TimeSequence<double>>;
    static constexpr Options::String help{"Times to force steps at"};
  };

  static constexpr Options::String help =
      "Suggests step sizes to place steps at specific times.\n"
      "\n"
      "The suggestion provided depends on the current time, so it should\n"
      "be applied immediately, rather than delayed several slabs.";
  using options = tmpl::list<Times>;

  explicit StepToTimes(std::unique_ptr<TimeSequence<double>> times)
      : times_(std::move(times)) {}

  using argument_tags = tmpl::list<::Tags::TimeStepId>;

  std::pair<double, bool> operator()(const TimeStepId& time_step_id,
                                     const double last_step_magnitude) const {
    const double now = time_step_id.substep_time();
    // Trying to step to a given time might not get us exactly there
    // because of rounding errors.  Avoid taking an extra tiny step if
    // we undershoot.
    const double sloppiness = slab_rounding_error(time_step_id.step_time());

    const auto goal_times = times_->times_near(now);
    if (not goal_times[1]) {
      // No times requested.
      return std::make_pair(std::numeric_limits<double>::infinity(), true);
    }

    double distance_to_next_goal = std::numeric_limits<double>::signaling_NaN();
    if (time_step_id.time_runs_forward()) {
      const auto next_time =
          *goal_times[1] > now + sloppiness ? goal_times[1] : goal_times[2];
      if (not next_time) {
        // We've passed all the times.  No restriction.
        return std::make_pair(std::numeric_limits<double>::infinity(), true);
      }
      distance_to_next_goal = *next_time - now;
    } else {
      const auto next_time =
          *goal_times[1] < now - sloppiness ? goal_times[1] : goal_times[0];
      if (not next_time) {
        // We've passed all the times.  No restriction.
        return std::make_pair(std::numeric_limits<double>::infinity(), true);
      }
      distance_to_next_goal = now - *next_time;
    }

    if (distance_to_next_goal < 2.0 / 3.0 * last_step_magnitude) {
      // Our goal is well within range of the expected allowed step
      // size.
      return std::make_pair(distance_to_next_goal, true);
    } else {
      // We can't reach our goal in one step, or at least might not be
      // able to if the step adjusts a relatively small amount for
      // other reasons.  Prevent the step from bringing us too close
      // to the goal so that the step following this one will not be
      // too small.
      return std::make_pair(2.0 / 3.0 * distance_to_next_goal, true);
    }
  }

  bool uses_local_data() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | times_; }

 private:
  std::unique_ptr<TimeSequence<double>> times_;
};
}  // namespace StepChoosers
