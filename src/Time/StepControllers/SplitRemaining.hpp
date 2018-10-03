// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class StepControllers::SplitRemaining

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/StepController.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

namespace StepControllers {

/// \ingroup TimeSteppersGroup
///
/// A StepController that chooses steps to be 1/n of the remainder of
/// the slab.
class SplitRemaining : public StepController {
 public:
  /// \cond
  SplitRemaining() = default;
  explicit SplitRemaining(CkMigrateMessage* /*unused*/) noexcept {}
  WRAPPED_PUPable_decl(SplitRemaining);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Chooses steps by dividing the remainder of the slab evenly.\n"
      "WARNING: With many steps per slab this often leads to overflow in the\n"
      "  time representations."};

  TimeDelta choose_step(const Time& time,
                        const double desired_step) const noexcept override {
    const Time goal =
        desired_step > 0 ? time.slab().end() : time.slab().start();
    const TimeDelta remaining = goal - time;
    const auto steps = std::max(
        static_cast<int32_t>(std::ceil(remaining.value() / desired_step)), 1);

    return remaining / steps;
  }
};
}  // namespace StepControllers
