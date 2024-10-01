// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <optional>
#include <utility>

#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/Utilities.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace StepChoosers {
/// Limits the time step to prevent multistep integrator instabilities.
///
/// Avoids instabilities due to rapid increases in the step size by
/// preventing the step size from increasing if any step in the
/// time-stepper history increased.  If there have been recent step
/// size increases, the new size bound is the size of the most recent
/// step, otherwise no restriction is imposed.
class PreventRapidIncrease : public StepChooser<StepChooserUse::Slab>,
                             public StepChooser<StepChooserUse::LtsStep> {
 public:
  /// \cond
  PreventRapidIncrease() = default;
  explicit PreventRapidIncrease(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PreventRapidIncrease);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
      "Limits the time step to prevent multistep integrator instabilities."};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<::Tags::HistoryEvolvedVariables<>>;

  template <typename History>
  std::pair<TimeStepRequest, bool> operator()(const History& history,
                                              const double last_step) const {
    if (history.size() < 2) {
      return {{}, true};
    }

    const double sloppiness =
        slab_rounding_error(history.front().time_step_id.step_time());
    std::optional<Time> previous_time{};
    double newer_step = abs(last_step);
    for (auto record = history.rbegin(); record != history.rend(); ++record) {
      const Time time = record->time_step_id.step_time();
      if (previous_time.has_value()) {
        const double this_step = abs(*previous_time - time).value();
        // Potential roundoff error comes from the inability to make
        // slabs exactly the same length.
        if (this_step < newer_step - sloppiness) {
          return {{.size = last_step}, true};
        }
        newer_step = this_step;
      }
      previous_time.emplace(time);
    }
    return {{}, true};
  }

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  void pup(PUP::er& p) override;
};
}  // namespace StepChoosers
