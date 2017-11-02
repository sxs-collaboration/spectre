// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class StepControllers::SimpleTimes

#pragma once

#include <cmath>

#include "Options/Options.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FractionUtilities.hpp"

namespace StepControllers {

/// \ingroup TimeSteppersGroup
///
/// A StepController that roughly splits the remaining time, but
/// prefers simpler (smaller denominator) fractions of slabs.
class SimpleTimes : public StepController {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString_t help = {
      "Chooses steps by dividing the remainder of the slab approximately\n"
      "evenly, but preferring evaluation times that are simple (i.e., small\n"
      "denominator) fractions of the slab."};

  TimeDelta choose_step(const Time& time,
                        const double desired_step) const noexcept override {
    const Time goal =
        desired_step > 0 ? time.slab().end() : time.slab().start();
    const TimeDelta remaining = goal - time;
    if (std::abs(desired_step) >= std::abs(remaining.value())) {
      return remaining;
    }

    // First arg: don't take a step so small that it will force an extra step
    // Second arg: don't choose a value too far from the request
    const double min_step =
        max_by_magnitude(std::fmod(remaining.value(), desired_step),
                         0.5 * desired_step);

    const double full_slab = time.slab().duration().value();
    const auto current_position =
        boost::rational_cast<double>(time.fraction());
    using Fraction = Time::rational_t;
    Fraction step_end = simplest_fraction_in_interval<Fraction>(
        current_position + min_step / full_slab,
        current_position + desired_step / full_slab);

    // clang-tidy: move trivially copyable type
    return Time(time.slab(), std::move(step_end)) - time; // NOLINT
  }
};
}  // namespace StepControllers
