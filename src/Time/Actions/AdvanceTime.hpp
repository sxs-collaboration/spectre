// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
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
/// Uses:
/// - DataBox:
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::Time
///   - Tags::TimeStepId
///   - Tags::TimeStep
struct AdvanceTime {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               Tags::Time>(
        make_not_null(&box),
        [](const gsl::not_null<TimeStepId*> time_id,
           const gsl::not_null<TimeStepId*> next_time_id,
           const gsl::not_null<TimeDelta*> time_step,
           const gsl::not_null<double*> time,
           const TimeStepper& time_stepper) noexcept {
          *time_id = *next_time_id;
          *time_step = time_step->with_slab(time_id->step_time().slab());
          *next_time_id = time_stepper.next_time_id(*next_time_id, *time_step);
          *time = time_id->substep_time().value();
        },
        db::get<Tags::TimeStepper<>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
