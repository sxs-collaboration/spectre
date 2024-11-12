// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/SelfStart.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Triggers/OnSubsteps.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace evolution::Actions {
namespace detail {
template <typename DbTags, typename Metavariables, typename ArrayIndex,
          typename ParallelComponent, typename EventsAndTriggers_t>
void run_events_and_triggers(db::DataBox<DbTags>& box,
                             Parallel::GlobalCache<Metavariables>& cache,
                             const ArrayIndex& array_index,
                             const ParallelComponent* const component,
                             const EventsAndTriggers_t& events_and_triggers,
                             const TimeStepId& time_step_id) {
  if (time_step_id.substep() == 0) {
    events_and_triggers.run_events(
        make_not_null(&box), cache, array_index, component,
        {db::tag_name<::Tags::Time>(), db::get<::Tags::Time>(box)});
  } else {
    const double substep_offset = 1.0e6;
    const double observation_value =
        time_step_id.step_time().value() +
        substep_offset * static_cast<double>(time_step_id.substep());
    events_and_triggers.run_events(
        make_not_null(&box), cache, array_index, component,
        {db::tag_name<::Tags::Time>(), observation_value},
        [&box](const Trigger& trigger) {
          const auto* substep_trigger =
              dynamic_cast<const ::Triggers::OnSubsteps*>(&trigger);
          return substep_trigger != nullptr and
                 substep_trigger->is_triggered(box);
        });
  }
}
}  // namespace detail

/// \ingroup ActionsGroup
/// \ingroup EventsAndTriggersGroup
/// \brief Run the events and triggers
///
/// Triggers will only be checked on the first step of each slab to
/// ensure that a consistent set of events is run across all elements.
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
template <bool LocalTimeStepping>
struct RunEventsAndTriggers {
  static constexpr bool local_time_stepping = LocalTimeStepping;
  using const_global_cache_tags = tmpl::conditional_t<
      local_time_stepping,
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
                 ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>,
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    const auto time_step_id = db::get<::Tags::TimeStepId>(box);
    if (SelfStart::is_self_starting(time_step_id)) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    if constexpr (local_time_stepping) {
      const auto& events_and_triggers_at_steps = Parallel::get<
          ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>(cache);
      detail::run_events_and_triggers(box, cache, array_index, component,
                                      events_and_triggers_at_steps,
                                      time_step_id);
    }

    if (not time_step_id.step_time().is_at_slab_boundary()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const auto& events_and_triggers_at_slabs = Parallel::get<
        ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(cache);
    detail::run_events_and_triggers(box, cache, array_index, component,
                                    events_and_triggers_at_slabs, time_step_id);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::Actions
