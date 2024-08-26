// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
struct AllStepChoosers;
class TimeDelta;
class TimeStepId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace StepChooserUse {
struct LtsStep;
}  // namespace StepChooserUse
namespace Tags {
template <typename Tag>
struct Next;
struct StepChoosers;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags

/// \endcond

/// \brief Adjust the step size for local time stepping, returning true if the
/// step just completed is accepted, and false if it is rejected.
///
/// \details The optional template parameter `StepChoosersToUse` may be used to
/// indicate a subset of the constructable step choosers to use for the current
/// application of `ChangeStepSize`. Passing `AllStepChoosers` (default)
/// indicates that any constructible step chooser may be used. This option is
/// used when multiple components need to invoke `ChangeStepSize` with step
/// choosers that may not be compatible with all components.
template <typename StepChoosersToUse = AllStepChoosers, typename DbTags>
bool change_step_size(const gsl::not_null<db::DataBox<DbTags>*> box) {
  const LtsTimeStepper& time_stepper =
      db::get<Tags::TimeStepper<LtsTimeStepper>>(*box);
  const auto& step_choosers = db::get<Tags::StepChoosers>(*box);

  const auto& time_step_id = db::get<Tags::TimeStepId>(*box);
  ASSERT(time_step_id.substep() == 0, "Can't change step size on a substep.");

  const auto current_step = db::get<Tags::TimeStep>(*box);
  const double last_step_size = current_step.value();

  TimeStepRequestProcessor step_requests(time_step_id.time_runs_forward());
  bool step_accepted = true;
  for (const auto& step_chooser : step_choosers) {
    const auto [step_request, step_choice_accepted] =
        step_chooser->template desired_step<StepChoosersToUse>(last_step_size,
                                                               *box);
    step_requests.process(step_request);
    step_accepted = step_accepted and step_choice_accepted;
  }

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
  if (not can_change_step_size) {
    step_requests.error_on_hard_limit(
        current_step.value(),
        (time_step_id.step_time() + current_step).value());
    return true;
  }

  const double desired_step = step_requests.step_size(
      time_step_id.step_time().value(), current_step.value());

  constexpr double smallest_relative_step_size = 1.0e-9;
  if (abs(desired_step / current_step.slab().duration().value()) <
      smallest_relative_step_size) {
    ERROR(
        "Chosen step is extremely small; this can indicate a flaw in the "
        "step chooser, the grid, or a simualtion instability that an "
        "error-based stepper is naively attempting to resolve. It is unlikely "
        "that the simulation can proceed in a stable and accurate manner. "
        "A possible issue is an aliasing-driven instability that could be "
        "cured by more aggressive filtering if you are using DG. The desired "
        "time step size is "
        << desired_step << " which, when divided by the slab size ("
        << current_step.slab().duration().value() << ") is below the tolerance "
        << smallest_relative_step_size << ". The current time is "
        << time_step_id.substep_time() << ".");
  }

  const auto new_step = choose_lts_step_size(
      time_step_id.step_time() + current_step, desired_step);
  db::mutate<Tags::Next<Tags::TimeStep>>(
      [&new_step](const gsl::not_null<TimeDelta*> next_step) {
        *next_step = new_step;
      },
      box);
  // if step accepted, just proceed. Otherwise, change Time::Next and jump
  // back to the first instance of `UpdateU`.
  if (step_accepted) {
    step_requests.error_on_hard_limit(
        current_step.value(),
        (time_step_id.step_time() + current_step).value());
    return true;
  } else {
    db::mutate<Tags::Next<Tags::TimeStepId>, Tags::TimeStep>(
        [&](const gsl::not_null<TimeStepId*> local_next_time_id,
            const gsl::not_null<TimeDelta*> time_step) {
          *time_step =
              choose_lts_step_size(time_step_id.step_time(), desired_step);
          ASSERT(*time_step != current_step,
                 "Step was rejected, but not changed."
                     << "\ntime_step_id = " << time_step_id
                     << "\ndesired_step = " << desired_step
                     << "\ntime_step = " << *time_step);
          *local_next_time_id =
              time_stepper.next_time_id(time_step_id, *time_step);
        },
        box);
    return false;
  }
}

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Adjust the step size for local time stepping
///
/// \details The optional template parameter `StepChoosersToUse` may be used to
/// indicate a subset of the constructable step choosers to use for the current
/// application of `ChangeStepSize`. Passing `AllStepChoosers` (default)
/// indicates that any constructible step chooser may be used. This option is
/// used when multiple components need to invoke `ChangeStepSize` with step
/// choosers that may not be compatible with all components.
///
/// Uses:
/// - DataBox:
///   - Tags::StepChoosers
///   - Tags::HistoryEvolvedVariables
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<LtsTimeStepper>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::Next<Tags::TimeStepId>, Tags::TimeStep
template <typename StepChoosersToUse = AllStepChoosers>
struct ChangeStepSize {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        tmpl::any<ActionList, tt::is_a<Actions::UpdateU, tmpl::_1>>::value,
        "The ChangeStepSize action requires that you also use the UpdateU "
        "action to permit step-unwinding. If you are stepping within "
        "an action that is not UpdateU, consider using the take_step function "
        "to handle both stepping and step-choosing instead of the "
        "ChangeStepSize action.");
    if (db::get<Tags::TimeStepId>(box).substep() != 0) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const bool step_successful =
        change_step_size<StepChoosersToUse>(make_not_null(&box));
    // We should update
    // AdaptiveSteppingDiagnostics::number_of_step_fraction_changes,
    // but with the inter-action step unwinding it's hard to tell
    // whether that happened.  Most executables use take_step instead
    // of this action, anyway.
    if (step_successful) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    } else {
      db::mutate<Tags::AdaptiveSteppingDiagnostics>(
          [](const gsl::not_null<AdaptiveSteppingDiagnostics*> diags) {
            ++diags->number_of_step_rejections;
          },
          make_not_null(&box));
      return {Parallel::AlgorithmExecution::Continue,
              tmpl::index_if<ActionList,
                             tt::is_a<Actions::UpdateU, tmpl::_1>>::value};
    }
  }
};
}  // namespace Actions
