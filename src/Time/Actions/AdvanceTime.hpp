// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
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
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
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
///   - Tags::TimeStepper<>
///
/// DataBox changes:
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::Next<Tags::TimeStep>
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
               Tags::AdaptiveSteppingDiagnostics>(
        [](const gsl::not_null<TimeStepId*> time_id,
           const gsl::not_null<TimeStepId*> next_time_id,
           const gsl::not_null<TimeDelta*> time_step,
           const gsl::not_null<double*> time,
           const gsl::not_null<TimeDelta*> next_time_step,
           const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
           const TimeStepper& time_stepper, const bool using_error_control) {
          if (time_id->slab_number() != next_time_id->slab_number()) {
            ++diags->number_of_slabs;
            // Put this here instead of unconditionally doing the next
            // check because on the first call time_id doesn't have a
            // valid slab so comparing the times will FPE.
            ++diags->number_of_steps;
          } else if (time_id->step_time() != next_time_id->step_time()) {
            ++diags->number_of_steps;
          }
          if (time_step->fraction() != next_time_step->fraction()) {
            ++diags->number_of_step_fraction_changes;
          }

          *time_id = *next_time_id;
          *time_step = next_time_step->with_slab(time_id->step_time().slab());

          if (using_error_control) {
            *next_time_id =
                time_stepper.next_time_id_for_error(*next_time_id, *time_step);
          } else {
            *next_time_id =
                time_stepper.next_time_id(*next_time_id, *time_step);
          }
          *next_time_step =
              time_step->with_slab(next_time_id->step_time().slab());
          *time = time_id->substep_time();
        },
        make_not_null(&box), db::get<Tags::TimeStepper<>>(box),
        is_using_error_control);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
