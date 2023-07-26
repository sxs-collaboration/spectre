// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Convergence::Tags {
template <typename Label>
struct IterationId;
}  // namespace Convergence::Tags
namespace Tags {
struct Time;
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup EventsAndTriggersGroup
/// \brief Run the events and triggers
///
/// Uses:
/// - GlobalCache: the EventsAndTriggers tag, as required by
///   events and triggers
/// - DataBox: as required by events and triggers
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename ObservationId>
struct RunEventsAndTriggers {
  static_assert(std::is_same_v<ObservationId, Tags::Time> or
                tt::is_a_v<Convergence::Tags::IterationId, ObservationId>);
  using const_global_cache_tags = tmpl::list<Tags::EventsAndTriggers>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    // Checking the slab status in this way will work for
    // non-evolutions and does not require linking against any
    // evolution-related libraries because everything is duck-typed
    // off the DataBox.
    if constexpr (std::is_same_v<ObservationId, Tags::Time>) {
      if (not db::get<Tags::TimeStepId>(box).is_at_slab_boundary()) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }
    }

    Parallel::get<Tags::EventsAndTriggers>(cache).run_events(
        box, cache, array_index, component,
        {db::tag_name<ObservationId>(),
         static_cast<double>(db::get<ObservationId>(box))});

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
