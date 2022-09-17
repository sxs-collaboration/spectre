// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Tags.hpp"
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
    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               Tags::Time, Tags::Next<Tags::TimeStep>>(
        make_not_null(&box),
        [](const gsl::not_null<TimeStepId*> time_id,
           const gsl::not_null<TimeStepId*> next_time_id,
           const gsl::not_null<TimeDelta*> time_step,
           const gsl::not_null<double*> time,
           const gsl::not_null<TimeDelta*> next_time_step,
           const TimeStepper& time_stepper) {
          *time_id = *next_time_id;
          *time_step = next_time_step->with_slab(time_id->step_time().slab());

          *next_time_id = time_stepper.next_time_id(*next_time_id, *time_step);
          *next_time_step =
              time_step->with_slab(next_time_id->step_time().slab());
          *time = time_id->substep_time().value();
        },
        db::get<Tags::TimeStepper<>>(box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
