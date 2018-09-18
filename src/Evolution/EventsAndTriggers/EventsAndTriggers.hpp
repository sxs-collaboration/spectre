// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Parallel/ConstGlobalCache.hpp"

/// \ingroup EventsAndTriggersGroup
/// Class that checks triggers and runs events
template <typename KnownEvents, typename KnownTriggers>
class EventsAndTriggers {
 public:
  using Storage =
      std::unordered_map<std::unique_ptr<Trigger<KnownTriggers>>,
                         std::vector<std::unique_ptr<Event<KnownEvents>>>>;

  EventsAndTriggers() = default;
  explicit EventsAndTriggers(Storage events_and_triggers) noexcept
      : events_and_triggers_(std::move(events_and_triggers)) {}

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename Component>
  void run_events(const db::DataBox<DbTags>& box,
                  Parallel::ConstGlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const Component* component) const noexcept {
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

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | events_and_triggers_;
  }

 private:
  // The unique pointer contents *must* be treated as const everywhere
  // in order to make the const global cache behave sanely.  They are
  // only non-const to make pup work.
  Storage events_and_triggers_;
};

template <typename KnownEvents, typename KnownTriggers>
struct create_from_yaml<EventsAndTriggers<KnownEvents, KnownTriggers>> {
  using type = EventsAndTriggers<KnownEvents, KnownTriggers>;
  static type create(const Option& options) {
    return type(options.parse_as<typename type::Storage>());
  }
};

namespace Tags {
/// \cond
struct EventsAndTriggersTagBase {};
/// \endcond

/// \ingroup OptionTagsGroup
/// \ingroup EventsAndTriggersGroup
/// Contains the events and triggers
template <typename KnownEvents, typename KnownTriggers>
struct EventsAndTriggers : EventsAndTriggersTagBase {
  using type = ::EventsAndTriggers<KnownEvents, KnownTriggers>;
};
}  // namespace Tags
