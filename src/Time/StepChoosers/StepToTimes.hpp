// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <pup.h>
#include <pup_stl.h>  // IWYU pragma: keep
#include <utility>
#include <vector>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Utilities.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
template <typename StepChooserRegistrars>
class StepToTimes;

namespace Registrars {
using StepToTimes = Registration::Registrar<StepChoosers::StepToTimes>;
}  // namespace Registrars

/// Suggests step sizes to place steps at specific times.
///
/// The suggestion provided depends on the current time, so it should
/// be applied immediately, rather than delayed several slabs.  As
/// changing immediately is inefficient, it may be best to use
/// triggers to only activate this check near (within a few slabs of)
/// the desired time.
template <typename StepChooserRegistrars = tmpl::list<Registrars::StepToTimes>>
class StepToTimes : public StepChooser<StepChooserRegistrars> {
 public:
  /// \cond
  StepToTimes() = default;
  explicit StepToTimes(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(StepToTimes);  // NOLINT
  /// \endcond

  struct Times {
    using type = std::vector<double>;
    static constexpr OptionString help{"Times to force steps at"};
  };

  static constexpr OptionString help =
      "Suggests step sizes to place steps at specific times.\n"
      "\n"
      "The suggestion provided depends on the current time, so it should\n"
      "be applied immediately, rather than delayed several slabs.  As\n"
      "changing immediately is inefficient, it may be best to use\n"
      "triggers to only activate this check near (within a few slabs of)\n"
      "the desired time.\n";
  using options = tmpl::list<Times>;

  explicit StepToTimes(std::vector<double> times) noexcept
      : times_(std::move(times)) {
    std::sort(times_.begin(), times_.end());
  }

  using argument_tags = tmpl::list<Tags::TimeStepId>;

  template <typename Metavariables>
  double operator()(
      const TimeStepId& time_step_id, const double last_step_magnitude,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const
      noexcept {
    const auto& substep_time = time_step_id.substep_time();
    const double now = substep_time.value();
    // Trying to step to a given time might not get us exactly there
    // because of rounding errors.  Avoid taking an extra tiny step if
    // we undershoot.
    const double sloppiness = slab_rounding_error(substep_time);

    double distance_to_next_goal = std::numeric_limits<double>::signaling_NaN();
    if (time_step_id.time_runs_forward()) {
      const auto next_time =
          std::upper_bound(times_.begin(), times_.end(), now + sloppiness);
      if (next_time == times_.end()) {
        // We've passed all the times.  No restriction.
        return std::numeric_limits<double>::infinity();
      }
      distance_to_next_goal = *next_time - now;
    } else {
      const auto next_time =
          std::upper_bound(times_.rbegin(), times_.rend(), now - sloppiness,
                           std::greater<double>{});
      if (next_time == times_.rend()) {
        // We've passed all the times.  No restriction.
        return std::numeric_limits<double>::infinity();
      }
      distance_to_next_goal = now - *next_time;
    }

    if (distance_to_next_goal < 2.0 / 3.0 * last_step_magnitude) {
      // Our goal is well within range of the expected allowed step
      // size.
      return distance_to_next_goal;
    } else {
      // We can't reach our goal in one step, or at least might not be
      // able to if the step adjusts a relatively small amount for
      // other reasons.  Prevent the step from bringing us too close
      // to the goal so that the step following this one will not be
      // too small.
      return 2.0 / 3.0 * distance_to_next_goal;
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | times_; }

 private:
  std::vector<double> times_;
};

/// \cond
template <typename StepChooserRegistrars>
PUP::able::PUP_ID StepToTimes<StepChooserRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
