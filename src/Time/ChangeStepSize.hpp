// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/MinimumTimeStep.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct AllStepChoosers;
namespace Tags {
struct AdaptiveSteppingDiagnostics;
struct DataBox;
struct FixedLtsRatio;
template <typename Tag>
struct Next;
struct StepChoosers;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
/// \endcond

/// \brief Adjust the step size for local time stepping.
///
/// \details
/// Usually, the new step size is chosen by calling the StepChoosers from
/// `Tags::StepChoosers`, restricted based on the allowed step sizes at the
/// current time, and limits from history initialization.
///
/// If `Tags::FixedLtsRatio` is present in the DataBox and not empty, the
/// StepChoosers are not called and instead the desired step is taken to be the
/// slab size over that value.  Early in the evolution, the actual chosen step
/// may differ from this because of restrictions on the allowed step, but all
/// such restrictions are global and will not result in different decisions for
/// different elements with the same desired fixed ratio.
///
/// The optional template parameter `StepChoosersToUse` may be used to
/// indicate a subset of the constructable step choosers to use for the current
/// application of `ChangeStepSize`. Passing `AllStepChoosers` (default)
/// indicates that any constructible step chooser may be used. This option is
/// used when multiple components need to invoke `ChangeStepSize` with step
/// choosers that may not be compatible with all components.
template <typename StepChoosersToUse = AllStepChoosers>
struct ChangeStepSize {
  using const_global_cache_tags = tmpl::list<Tags::MinimumTimeStep>;

  using return_tags = tmpl::list<Tags::DataBox>;
  using argument_tags = tmpl::list<>;

  template <typename DbTags>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box) {
    const auto& time_step_id = db::get<Tags::TimeStepId>(*box);
    if (time_step_id.substep() != 0) {
      return;
    }

    const LtsTimeStepper& time_stepper =
        db::get<Tags::TimeStepper<LtsTimeStepper>>(*box);
    const auto& step_choosers = db::get<Tags::StepChoosers>(*box);

    using history_tags = ::Tags::get_all_history_tags<DbTags>;
    bool can_change_step_size = true;
    tmpl::for_each<history_tags>([&box, &can_change_step_size, &time_stepper,
                                  &time_step_id](auto tag_v) {
      if (not can_change_step_size) {
        return;
      }
      using tag = typename decltype(tag_v)::type;
      const auto& history = db::get<tag>(*box);
      can_change_step_size =
          time_stepper.can_change_step_size(time_step_id, history);
    });

    const auto current_step = db::get<Tags::TimeStep>(*box);

    std::optional<size_t> fixed_lts_ratio{};
    if constexpr (db::tag_is_retrievable_v<Tags::FixedLtsRatio,
                                           db::DataBox<DbTags>>) {
      fixed_lts_ratio = db::get<Tags::FixedLtsRatio>(*box);
    }

    TimeStepRequestProcessor step_requests(time_step_id.time_runs_forward());
    if (fixed_lts_ratio.has_value()) {
      ASSERT(std::popcount(*fixed_lts_ratio) == 1,
             "fixed_lts_ratio must be a power of 2, not " << *fixed_lts_ratio);
      step_requests.process(TimeStepRequest{
          .size_goal =
              (current_step.slab().duration() / *fixed_lts_ratio).value()});
    } else {
      const double last_step_size = current_step.value();
      for (const auto& step_chooser : step_choosers) {
        const auto step_request =
            step_chooser->template desired_step<StepChoosersToUse>(
                last_step_size, *box);
        step_requests.process(step_request);
      }
    }

    if (not can_change_step_size) {
      step_requests.error_on_hard_limit(
          current_step.value(),
          (time_step_id.step_time() + current_step).value());
      return;
    }

    const double desired_step = step_requests.step_size(
        time_step_id.step_time().value(), current_step.value());

    // We do this check twice, first on the desired value, and then on
    // the actual chosen value, which is probably slightly smaller.
    if (std::abs(desired_step) < db::get<::Tags::MinimumTimeStep>(*box)) {
      ERROR_NO_TRACE(
          "Chosen step size "
          << desired_step << " is smaller than the MinimumTimeStep of "
          << db::get<::Tags::MinimumTimeStep>(*box)
          << ".\n"
             "\n"
             "This can indicate a flaw in the step chooser, the grid, or a "
             "simulation instability that an error-based stepper is naively "
             "attempting to resolve. A possible issue is an aliasing-driven "
             "instability that could be cured by more aggressive filtering if "
             "you are using DG.");
    }

    const auto new_step =
        choose_lts_step_size(time_step_id.step_time(), desired_step);
    step_requests.error_on_hard_limit(
        new_step.value(), (time_step_id.step_time() + new_step).value());

    if (new_step == current_step) {
      return;
    }

    if (std::abs(new_step.value()) < db::get<::Tags::MinimumTimeStep>(*box)) {
      ERROR_NO_TRACE(
          "Chosen step size after conversion to a fraction of a slab "
          << new_step << " is smaller than the MinimumTimeStep of "
          << db::get<::Tags::MinimumTimeStep>(*box)
          << ".\n"
             "\n"
             "This can indicate a flaw in the step chooser, the grid, or a "
             "simulation instability that an error-based stepper is naively "
             "attempting to resolve. A possible issue is an aliasing-driven "
             "instability that could be cured by more aggressive filtering if "
             "you are using DG.");
    }

    db::mutate<Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               Tags::AdaptiveSteppingDiagnostics>(
        [&](const gsl::not_null<TimeStepId*> local_next_time_id,
            const gsl::not_null<TimeDelta*> time_step,
            const gsl::not_null<AdaptiveSteppingDiagnostics*> diags) {
          *time_step = new_step;
          *local_next_time_id =
              time_stepper.next_time_id(time_step_id, *time_step);
          ++diags->number_of_step_fraction_changes;
        },
        box);
  }
};
