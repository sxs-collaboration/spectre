// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep  // for Tags::Next
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Tags.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
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
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
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
  const LtsTimeStepper& time_stepper = db::get<Tags::TimeStepper<>>(*box);
  const auto& step_choosers = db::get<Tags::StepChoosers>(*box);

  const auto& next_time_id = db::get<Tags::Next<Tags::TimeStepId>>(*box);
  using history_tags = ::Tags::get_all_history_tags<DbTags>;
  bool can_change_step_size = true;
  tmpl::for_each<history_tags>([&box, &can_change_step_size, &time_stepper,
                                &next_time_id](auto tag_v) {
    if (not can_change_step_size) {
      return;
    }
    using tag = typename decltype(tag_v)::type;
    const auto& history = db::get<tag>(*box);
    can_change_step_size =
        time_stepper.can_change_step_size(next_time_id, history);
  });
  if (not can_change_step_size) {
    return true;
  }

  const auto& current_step = db::get<Tags::TimeStep>(*box);

  const double last_step_size = std::abs(db::get<Tags::TimeStep>(*box).value());

  // The step choosers return the magnitude of the desired step, so
  // we always want the minimum requirement, but we have to negate
  // the final answer if time is running backwards.
  double desired_step = std::numeric_limits<double>::infinity();
  bool step_accepted = true;
  for (const auto& step_chooser : step_choosers) {
    const auto [step_choice, step_choice_accepted] =
        step_chooser->template desired_step<StepChoosersToUse>(
            last_step_size, *box);
    desired_step = std::min(desired_step, step_choice);
    step_accepted = step_accepted and step_choice_accepted;
  }
  if (not current_step.is_positive()) {
    desired_step = -desired_step;
  }

  if (abs(desired_step / current_step.slab().duration().value()) < 1.0e-9) {
    ERROR(
        "Chosen step is extremely small; this can indicate a flaw in the a "
        "step chooser, the grid, or a simualtion instability that an "
        "error-based stepper is naively attempting to resolve. It is unlikely "
        "that the simulation can proceed");
  }

  const auto new_step =
      choose_lts_step_size(next_time_id.step_time(), desired_step);
  db::mutate<Tags::Next<Tags::TimeStep>>(
      [&new_step](const gsl::not_null<TimeDelta*> next_step) {
        *next_step = new_step;
      },
      box);
  // if step accepted, just proceed. Otherwise, change Time::Next and jump
  // back to the first instance of `UpdateU`.
  if (step_accepted) {
    return true;
  } else {
    db::mutate<Tags::Next<Tags::TimeStepId>, Tags::TimeStep>(
        [&time_stepper, &desired_step](
            const gsl::not_null<TimeStepId*> local_next_time_id,
            const gsl::not_null<TimeDelta*> time_step,
            const TimeStepId& time_id) {
          *time_step = choose_lts_step_size(time_id.step_time(), desired_step);
          *local_next_time_id = time_stepper.next_time_id(time_id, *time_step);
        },
        box, db::get<Tags::TimeStepId>(*box));
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
///   - Tags::StepChoosers<StepChooserRegistrars>
///   - Tags::HistoryEvolvedVariables
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<>
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
