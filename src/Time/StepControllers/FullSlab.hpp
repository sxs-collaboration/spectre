// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class StepControllers::FullSlab

#pragma once

#include <pup.h>

#include "ErrorHandling/Assert.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/StepController.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

namespace StepControllers {

/// \ingroup TimeSteppersGroup
///
/// A StepController that always chooses to step a full slab,
/// independent of the desired step size.  Intended for simuations
/// using global time stepping.
class FullSlab : public StepController {
 public:
  /// \cond
  FullSlab() = default;
  explicit FullSlab(CkMigrateMessage* /*unused*/) noexcept {}
  WRAPPED_PUPable_decl(FullSlab);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help = {"Chooses the full slab."};

  TimeDelta choose_step(const Time& time,
                        const double desired_step) const noexcept override {
    ASSERT(time.is_at_slab_boundary(),
           "Trying to take a full slab step from the middle of a slab.");
    return desired_step > 0 ? time.slab().duration() : -time.slab().duration();
  }
};
}  // namespace StepControllers
