// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class StepControllers::BinaryFraction

#pragma once

#include <cmath>

#include "Options/Options.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace StepControllers {

/// \ingroup TimeSteppersGroup
///
/// A StepController that chooses steps to be 1/2^n of a slab.
class BinaryFraction : public StepController {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Chooses steps to be binary fractions of a slab"};

  TimeDelta choose_step(const Time& time,
                        const double desired_step) const noexcept override {
    ASSERT((time.fraction().denominator() &
            (time.fraction().denominator() - 1)) == 0,
           "Not at a binary-fraction time within slab: " << time.fraction());
    const TimeDelta full_slab =
        desired_step > 0 ? time.slab().duration() : -time.slab().duration();
    const double desired_step_count = full_slab.value() / desired_step;
    const size_t desired_step_power =
        desired_step_count <= 1
            ? 0
            : static_cast<size_t>(std::ceil(std::log2(desired_step_count)));

    // Ensure we will hit the slab boundary if we continue taking
    // constant-sized steps.
    const ssize_t step_count =
        std::max(static_cast<ssize_t>(two_to_the(desired_step_power)),
                 time.fraction().denominator());

    return full_slab / step_count;
  }
};
}  // namespace StepControllers
