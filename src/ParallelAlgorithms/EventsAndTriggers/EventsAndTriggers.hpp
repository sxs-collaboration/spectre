// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>  // IWYU pragma: keep
#include <unordered_map>
#include <vector>

#include "Options/Options.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

/// \ingroup EventsAndTriggersGroup
/// Class that checks triggers and runs events
class EventsAndTriggers {
 public:
  using Storage = std::unordered_map<std::unique_ptr<Trigger>,
                                     std::vector<std::unique_ptr<Event>>>;

  EventsAndTriggers() = default;
  explicit EventsAndTriggers(Storage events_and_triggers)
      : events_and_triggers_(std::move(events_and_triggers)) {}

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename Component>
  void run_events(const db::DataBox<DbTags>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const Component* component) const {
    for (const auto& trigger_and_events : events_and_triggers_) {
      const auto& trigger = trigger_and_events.first;
      const auto& events = trigger_and_events.second;
      if (trigger->is_triggered(box)) {
        for (const auto& event : events) {
          event->run(box, cache, array_index, component);
        }
      }
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | events_and_triggers_; }

  template <typename F>
  void for_each_event(F&& f) const {
    for (const auto& trigger_and_events : events_and_triggers_) {
      for (const auto& event : trigger_and_events.second) {
        f(*event);
      }
    }
  }

 private:
  // The unique pointer contents *must* be treated as const everywhere
  // in order to make the const global cache behave sanely.  They are
  // only non-const to make pup work.
  Storage events_and_triggers_;
};

template <>
struct Options::create_from_yaml<EventsAndTriggers> {
  using type = EventsAndTriggers;
  template <typename Metavariables>
  static type create(const Options::Option& options) {
    return type(options.parse_as<typename type::Storage, Metavariables>());
  }
};
