// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
 private:
  template <typename Event>
  struct get_tags {
    using type = typename Event::compute_tags_for_observation_box;
  };

 public:
  struct TriggerAndEvents {
    struct Trigger {
      using type = std::unique_ptr<::Trigger>;
      static constexpr Options::String help = "Determines when the Events run.";
    };
    struct Events {
      using type = std::vector<std::unique_ptr<::Event>>;
      static constexpr Options::String help =
          "These events run when the Trigger fires.";
    };
    static constexpr Options::String help =
        "Events that run when the Trigger fires.";
    using options = tmpl::list<Trigger, Events>;
    void pup(PUP::er& p) {
      p | trigger;
      p | events;
    }
    std::unique_ptr<::Trigger> trigger;
    std::vector<std::unique_ptr<::Event>> events;
  };

  using Storage = std::vector<TriggerAndEvents>;

  EventsAndTriggers() = default;
  explicit EventsAndTriggers(Storage events_and_triggers)
      : events_and_triggers_(std::move(events_and_triggers)) {}

  /// Check the triggers and run the associated events.
  ///
  /// By default the trigger check just calls the `is_triggered`
  /// method, but the last argument can be passed to override this.
  /// It must be a functor taking a `const Trigger&` and returning
  /// `bool`.
  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename Component, typename CheckTrigger = std::nullptr_t>
  void run_events(const gsl::not_null<db::DataBox<DbTags>*> box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index, const Component* component,
                  const Event::ObservationValue& observation_value,
                  const CheckTrigger& check_trigger = nullptr) const {
    using compute_tags = tmpl::remove_duplicates<tmpl::filter<
        tmpl::flatten<tmpl::transform<
            tmpl::at<typename Metavariables::factory_creation::factory_classes,
                     Event>,
            get_tags<tmpl::_1>>>,
        db::is_compute_tag<tmpl::_1>>>;
    std::optional<decltype(make_observation_box<compute_tags>(box))>
        observation_box{};
    for (const auto& trigger_and_events : events_and_triggers_) {
      const auto& trigger = trigger_and_events.trigger;
      const auto& events = trigger_and_events.events;
      const bool is_triggered = [&]() {
        if constexpr (std::is_same_v<std::decay_t<CheckTrigger>,
                                     std::nullptr_t>) {
          return trigger->is_triggered(*box);
        } else {
          return check_trigger(std::as_const(*trigger));
        }
      }();
      if (is_triggered) {
        if (not observation_box.has_value()) {
          observation_box = make_observation_box<compute_tags>(box);
        }
        for (const auto& event : events) {
          event->run(make_not_null(&observation_box.value()), cache,
                     array_index, component, observation_value);
        }
      }
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | events_and_triggers_; }

  template <typename F>
  void for_each_event(F&& f) const {
    for (const auto& trigger_and_events : events_and_triggers_) {
      for (const auto& event : trigger_and_events.events) {
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
