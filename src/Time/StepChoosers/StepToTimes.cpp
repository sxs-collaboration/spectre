// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/StepToTimes.hpp"

#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Time/EvolutionOrdering.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Gsl.hpp"

namespace StepChoosers {
StepToTimes::StepToTimes(std::unique_ptr<TimeSequence<double>> times)
    : times_(std::move(times)) {}

std::pair<TimeStepRequest, bool> StepToTimes::operator()(
    const double now, const double last_step) const {
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

bool StepToTimes::uses_local_data() const { return false; }
bool StepToTimes::can_be_delayed() const { return false; }

void StepToTimes::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  p | times_;
}

PUP::able::PUP_ID StepToTimes::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers
