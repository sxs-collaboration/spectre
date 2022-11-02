// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>  // IWYU pragma: keep  // for abs
#include <limits>
#include <pup.h>
#include <utility>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Tags.hpp"
#include "Time/Utilities.hpp"
#include "Utilities/TMPL.hpp"

namespace StepChoosers {
/// Avoids instabilities due to rapid increases in the step size by
/// preventing the step size from increasing unless all steps in the
/// time-stepper history are the same size.  If there have been recent
/// step size changes the new size bound is the size of the most
/// recent step, otherwise it is infinite (no restriction is imposed).
template <typename StepChooserUse>
class PreventRapidIncrease : public StepChooser<StepChooserUse> {
 public:
  /// \cond
  PreventRapidIncrease() = default;
  explicit PreventRapidIncrease(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PreventRapidIncrease);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
      "Prevents rapid increases in time step that can cause integrator \n"
      "instabilities."};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<::Tags::HistoryEvolvedVariables<>>;
  using return_tags = tmpl::list<>;

  template <typename History>
  std::pair<double, bool> operator()(const History& history,
                                     const double last_step_magnitude) const {
    if (history.size() < 2) {
      return std::make_pair(std::numeric_limits<double>::infinity(), true);
    }

    const double sloppiness = slab_rounding_error(history[0]);
    std::optional<Time> history_step{};
    for (auto step = history.begin(); step != history.end(); ++step) {
      if (step.time_step_id().substep() != 0) {
        continue;
      }
      if (history_step.has_value()) {
        // Potential roundoff error comes from the inability to make
        // slabs exactly the same length.
        if (abs(abs(*history_step - *step).value() - last_step_magnitude) >
            sloppiness) {
          return std::make_pair(last_step_magnitude, true);
        }
      }
      history_step.emplace(*step);
    }
    // Request that the step size be at most infinity.  This imposes
    // no restriction on the chosen step.
    return std::make_pair(std::numeric_limits<double>::infinity(), true);
  }

  bool uses_local_data() const override { return false; }
};

/// \cond
template <typename StepChooserUse>
PUP::able::PUP_ID PreventRapidIncrease<StepChooserUse>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
