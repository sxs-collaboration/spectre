// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>  // IWYU pragma: keep  // for abs
#include <limits>
#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Tags.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class Time;
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace StepChoosers {
template <typename StepChooserRegistrars>
class PreventRapidIncrease;

namespace Registrars {
using PreventRapidIncrease =
    Registration::Registrar<StepChoosers::PreventRapidIncrease>;
}  // namespace Registrars

/// Avoids instabilities due to rapid increases in the step size by
/// preventing the step size from increasing unless all steps in the
/// time-stepper history are the same size.  If there have been recent
/// step size changes the new size bound is the size of the most
/// recent step, otherwise it is infinite (no restriction is imposed).
///
/// Changes in step size resulting from a slab size change are not
/// taken into account.  In practice, this should not be an issue as
/// long as there are many steps between slab size changes.
template <typename StepChooserRegistrars =
              tmpl::list<Registrars::PreventRapidIncrease>>
class PreventRapidIncrease : public StepChooser<StepChooserRegistrars> {
 public:
  /// \cond
  PreventRapidIncrease() = default;
  explicit PreventRapidIncrease(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PreventRapidIncrease);  // NOLINT
  /// \endcond

  static constexpr OptionString help{
      "Prevents rapid increases in time step that can cause integrator \n"
      "instabilities."};
  using options = tmpl::list<>;

  using argument_tags =
      tmpl::list<Tags::SubstepTime, Tags::HistoryEvolvedVariables<>>;

  template <typename Metavariables, typename History>
  double operator()(const Time& time, const History& history,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/)
      const noexcept {
    if (history.size() < 2) {
      return std::numeric_limits<double>::infinity();
    }

    const auto last_step = time - history[history.size() - 1];

    // Slab boundaries are complicated, so we'll just ignore the slab
    // information and assume slab sizes don't change too frequently.
    // An occasional double increase should not be harmful.
    for (auto step = history.begin(); step != history.end() - 1; ++step) {
      if ((*(step + 1) - *step).fraction() != last_step.fraction()) {
        return abs(last_step.value());
      }
    }
    // Request that the step size be at most infinity.  This imposes
    // no restriction on the chosen step.
    return std::numeric_limits<double>::infinity();
  }
};

/// \cond
template <typename StepChooserRegistrars>
PUP::able::PUP_ID PreventRapidIncrease<StepChooserRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
