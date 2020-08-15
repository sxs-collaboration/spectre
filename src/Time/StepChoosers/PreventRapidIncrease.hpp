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
#include "Time/Utilities.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class Time;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
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

  using argument_tags = tmpl::list<Tags::HistoryEvolvedVariables<>>;

  template <typename Metavariables, typename History>
  double operator()(const History& history, const double last_step_magnitude,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/)
      const noexcept {
    if (history.size() < 2) {
      return std::numeric_limits<double>::infinity();
    }

    const double sloppiness = slab_rounding_error(history[0]);
    for (auto step = history.begin(); step != history.end() - 1; ++step) {
      // Potential roundoff error comes from the inability to make
      // slabs exactly the same length.
      if (abs(abs(*(step + 1) - *step).value() - last_step_magnitude) >
          sloppiness) {
        return last_step_magnitude;
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
