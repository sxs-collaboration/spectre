// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Options/String.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
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

  using argument_tags = tmpl::list<::Tags::Time>;

  std::pair<TimeStepRequest, bool> operator()(const double now,
                                              const double last_step) const {
    const auto goal_times = times_->times_near(now);
    if (not goal_times[1].has_value()) {
      // No times requested.
      return {{}, true};
    }

    const evolution_greater<double> after{last_step > 0.0};
    const auto next_time = after(*goal_times[1], now)
                               ? goal_times[1]
                               : gsl::at(goal_times, last_step > 0.0 ? 2 : 0);
    if (not next_time.has_value()) {
      // We've passed all the times.  No restriction.
      return {{}, true};
    }

    // The calling code can ignore one part of the request if it can
    // fulfill the other part exactly, so this will either step
    // exactly to *next_time or step at most 2/3 of the way there.
    // This attempts to balance out the steps to avoid a step almost
    // to *next_time followed by a very small step.  This will work
    // poorly if there are two copies of this StepChooser targeting
    // times very close together, but hopefully people don't do that.
    return {{.size = 2.0 / 3.0 * (*next_time - now),
             .end = *next_time,
             .end_hard_limit = *next_time},
            true};
  }

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | times_; }

 private:
  std::unique_ptr<TimeSequence<double>> times_;
};
}  // namespace StepChoosers
