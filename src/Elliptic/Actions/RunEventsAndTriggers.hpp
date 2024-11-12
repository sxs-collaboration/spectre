// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::Actions {
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
  using const_global_cache_tags =
      tmpl::list<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtIterations>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    Parallel::get<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtIterations>>(
        cache)
        .run_events(make_not_null(&box), cache, array_index, component,
                    {db::tag_name<ObservationId>(),
                     static_cast<double>(db::get<ObservationId>(box))});

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace elliptic::Actions
