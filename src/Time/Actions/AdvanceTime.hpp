// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct IsUsingTimeSteppingErrorControl;
template <typename Tag>
struct Next;
struct Time;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags

/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Advance time one substep
///
/// Replaces the time state with the `Tags::Next` values, advances the
/// `Tags::Next` values, and sets `Tags::Time` to the new substep time.
///
/// Uses:
/// - DataBox:
///   - Tags::Next<Tags::TimeStep>
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::TimeStepper<TimeStepper>
///
/// DataBox changes:
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::Next<Tags::TimeStep>
///   - Tags::StepNumberWithinSlab
///   - Tags::Time
///   - Tags::TimeStepId
///   - Tags::TimeStep
struct AdvanceTime {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    bool is_using_error_control = false;
    if constexpr (db::tag_is_retrievable_v<
                      Tags::IsUsingTimeSteppingErrorControl,
                      db::DataBox<DbTags>>) {
      is_using_error_control =
          db::get<Tags::IsUsingTimeSteppingErrorControl>(box);
    }

    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               Tags::Time, Tags::Next<Tags::TimeStep>,
               Tags::StepNumberWithinSlab, Tags::AdaptiveSteppingDiagnostics>(
        [](const gsl::not_null<TimeStepId*> time_id,
           const gsl::not_null<TimeStepId*> next_time_id,
           const gsl::not_null<TimeDelta*> time_step,
           const gsl::not_null<double*> time,
           const gsl::not_null<TimeDelta*> next_time_step,
           const gsl::not_null<uint64_t*> step_number_within_slab,
           const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
           const TimeStepper& time_stepper, const bool using_error_control) {
          const bool new_step = next_time_id->substep() == 0;
          if (time_id->slab_number() != next_time_id->slab_number()) {
            *step_number_within_slab = 0;
            ++diags->number_of_slabs;
            // Put this here instead of unconditionally doing the next
            // check because on the first call time_id doesn't have a
            // valid slab so comparing the times will FPE.
            ++diags->number_of_steps;
          } else if (new_step) {
            ++(*step_number_within_slab);
            ++diags->number_of_steps;
          }

          *time_id = *next_time_id;

          if (new_step) {
            if (time_step->fraction() != next_time_step->fraction()) {
              ++diags->number_of_step_fraction_changes;
            }

            *time_step = next_time_step->with_slab(time_id->step_time().slab());
            *next_time_step = *time_step;
          }

          if (using_error_control) {
            *next_time_id =
                time_stepper.next_time_id_for_error(*next_time_id, *time_step);
          } else {
            *next_time_id =
                time_stepper.next_time_id(*next_time_id, *time_step);
          }
          *time = time_id->substep_time();
        },
        make_not_null(&box), db::get<Tags::TimeStepper<TimeStepper>>(box),
        is_using_error_control);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
