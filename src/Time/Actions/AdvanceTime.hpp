// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TaggedTuple.hpp"

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
    db::mutate<Tags::TimeId, Tags::TimeStep>(
        box, [&cache](auto& time_id, auto& time_step) noexcept {
          const auto& time_stepper =
              Parallel::get<CacheTags::TimeStepper>(cache);
          time_id = time_stepper.next_time_id(time_id, time_step);
          if (time_id.is_at_slab_boundary() and
              (time_step.is_positive() ? time_id.time.is_at_slab_end()
                                       : time_id.time.is_at_slab_start())) {
            ++time_id.slab_number;
            const Slab new_slab =
                time_id.time.slab().advance_towards(time_step);
            time_id.time = time_id.time.with_slab(new_slab);
            time_step = time_step.with_slab(new_slab);
          }
        });

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
