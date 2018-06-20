// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace CacheTags {
struct TimeStepper;
}  // namespace CacheTags
namespace Tags {
template <typename Tag>
struct Next;
struct TimeId;
struct TimeStep;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Advance time one substep
///
/// Uses:
/// - ConstGlobalCache: CacheTags::TimeStepper
/// - DataBox: Tags::TimeId, Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::TimeId, Tags::TimeStep
struct AdvanceTime {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>(
        make_not_null(&box), [&cache](const gsl::not_null<TimeId*> time_id,
                                      const gsl::not_null<TimeId*> next_time_id,
                                      const gsl::not_null<TimeDelta*>
                                          time_step) noexcept {
          const auto& time_stepper =
              Parallel::get<CacheTags::TimeStepper>(cache);
          *time_id = *next_time_id;
          *time_step = time_step->with_slab(time_id->time().slab());
          *next_time_id = time_stepper.next_time_id(*next_time_id, *time_step);
          if (next_time_id->is_at_slab_boundary() and
              (next_time_id->time_runs_forward()
                   ? next_time_id->time().is_at_slab_end()
                   : next_time_id->time().is_at_slab_start())) {
            const Slab new_slab =
                next_time_id->time().slab().advance_towards(*time_step);
            *next_time_id =
                TimeId(next_time_id->time_runs_forward(),
                       next_time_id->slab_number() + 1,
                       next_time_id->step_time().with_slab(new_slab));
          }
        });

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
