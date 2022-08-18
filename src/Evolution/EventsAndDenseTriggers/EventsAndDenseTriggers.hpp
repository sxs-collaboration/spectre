// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct Time;
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace evolution {
namespace Tags {
/*!
 * \ingroup EventsAndTriggersGroup
 * \brief Previous time at which the trigger activated.
 *
 * \details This tag is populated with the most recent time the current trigger
 * fired prior to the current activation.
 * \note This tag is only populated with a meaningful value for events invoked
 * from `EventsAndDenseTriggers`; during other actions, the tag should not be
 * used.
 */
struct PreviousTriggerTime : db::SimpleTag {
  using type = std::optional<double>;
};
}  // namespace Tags

/// \ingroup EventsAndTriggersGroup
/// Class that checks dense triggers and runs events
class EventsAndDenseTriggers {
 private:
  template <typename Event>
  struct get_tags {
    using type = typename Event::compute_tags_for_observation_box;
  };

 public:
  using ConstructionType =
      std::unordered_map<std::unique_ptr<DenseTrigger>,
                         std::vector<std::unique_ptr<Event>>>;

 private:
  struct TriggerRecord {
    double next_check;
    std::unique_ptr<DenseTrigger> trigger;
    std::vector<std::unique_ptr<Event>> events;

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);
  };
  using Storage = std::vector<TriggerRecord>;

 public:
  EventsAndDenseTriggers() = default;
  explicit EventsAndDenseTriggers(ConstructionType events_and_triggers);

  template <typename DbTags>
  double next_trigger(const db::DataBox<DbTags>& box);

  enum class TriggeringState { Ready, NeedsEvolvedVariables, NotReady };

  /// Check triggers fire and whether all events to run are ready.  If
  /// this function returns anything other than NotReady, then
  /// rerunning it will skip checks (except for
  /// Event::needs_evolved_variables) until a successful call to
  /// reschedule.
  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  TriggeringState is_ready(const db::DataBox<DbTags>& box,
                           Parallel::GlobalCache<Metavariables>& cache,
                           const ArrayIndex& array_index,
                           const ComponentPointer component);

  /// Run events associated with fired triggers.  This must be called
  /// after is_ready returns something other then NotReady.  Any
  /// repeated calls will be no-ops until a successful call to
  /// reschedule.
  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  void run_events(db::DataBox<DbTags>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ComponentPointer component);

  /// Schedule the next check.  This must be called after run_events
  /// for the current check.  Returns `true` on success, `false` if
  /// insufficient data is available and the call should be retried
  /// later.
  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  bool reschedule(const db::DataBox<DbTags>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ComponentPointer component);

  /// Add a new trigger and set of events.  This can only be called
  /// during initialization.
  void add_trigger_and_events(std::unique_ptr<DenseTrigger> trigger,
                              std::vector<std::unique_ptr<Event>> events);

  template <typename F>
  void for_each_event(F&& f) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  typename Storage::iterator heap_end() {
    return events_and_triggers_.begin() +
           static_cast<Storage::difference_type>(heap_size_);
  }

  template <typename DbTags>
  void initialize(const db::DataBox<DbTags>& box);

  bool initialized() const;

  void populate_active_triggers();

  void reschedule_next_trigger(double next_check_time, bool time_runs_forward);

  class TriggerTimeAfter {
   public:
    explicit TriggerTimeAfter(const bool time_runs_forward);

    bool operator()(const TriggerRecord& a, const TriggerRecord& b) const;

    double infinite_future() const;

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);

   private:
    evolution_greater<double> time_after_{};
  };

  // The data structure used here is a heap containing the triggers
  // not being tested at the moment, with the earliest time as the
  // root at index 0, followed by the triggers being processed at the
  // current time.
  //
  // state at the start of a loop:
  // non-testing heap | to be tested
  // 0                  heap_size_ = to_run_position_ = processing_position_
  //
  // is_ready checks if everything is_ready and if triggers fire,
  // moving processing_position_ forward and swapping triggers into
  // the appropriate areas:
  //
  //                              ->      -->
  // non-testing heap | not fired | fired | to be tested
  // 0                  heap_size_  to_run_position_
  //                                        processing_position_
  //
  // run_events runs all events for triggers that have fired:
  //
  //                              ->
  // non-testing heap | not fired | fired
  // 0                  heap_size_  to_run_position_
  //
  // reschedule moves triggers back onto the heap...:
  //
  //                  ->
  // non-testing heap | triggers to reschedule
  // 0                  heap_size_
  //
  // ...and then sets up for the next loop by examining the heap root
  // and popping all triggers requesting that time into the processing
  // area.
  Storage events_and_triggers_;
  // The size of the heap, or equivalently the index of the start of
  // the processing area.  The initial value is a sentinel for an
  // uninitialized state.
  size_t heap_size_ = std::numeric_limits<size_t>::max();
  size_t to_run_position_{};
  size_t processing_position_{};
  // Index of the current event of the trigger at processing_position_
  // that we are waiting for to be ready, or none if we are waiting
  // for the trigger.
  std::optional<size_t> event_to_check_{};
  double next_check_ = std::numeric_limits<double>::signaling_NaN();
  TriggerTimeAfter next_check_after_{false};
};

template <typename DbTags>
double EventsAndDenseTriggers::next_trigger(const db::DataBox<DbTags>& box) {
  if (UNLIKELY(not initialized())) {
    initialize(box);
  }

  if (events_and_triggers_.empty()) {
    return next_check_after_.infinite_future();
  }

  return next_check_;
}

template <typename DbTags, typename Metavariables, typename ArrayIndex,
          typename ComponentPointer>
EventsAndDenseTriggers::TriggeringState EventsAndDenseTriggers::is_ready(
    const db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index, const ComponentPointer component) {
  ASSERT(initialized(), "Not initialized");
  ASSERT(not events_and_triggers_.empty(),
         "Should not be calling is_ready with no triggers");
  ASSERT(heap_end() != events_and_triggers_.end(), "No active triggers");

  for (; processing_position_ != events_and_triggers_.size();
       ++processing_position_) {
    const auto& current_trigger = events_and_triggers_[processing_position_];
    if (not event_to_check_.has_value()) {
      const auto is_triggered = current_trigger.trigger->is_triggered(
          box, cache, array_index, component);
      if (not is_triggered.has_value()) {
        return TriggeringState::NotReady;
      }

      if (not *is_triggered) {
        // Move this trigger into the non-fired area.  One of the
        // earlier fired triggers is moved here, but we don't make any
        // guarantees about the order anyway so that doesn't matter.
        std::swap(events_and_triggers_[processing_position_],
                  events_and_triggers_[to_run_position_]);
        ++to_run_position_;
        continue;
      }
      event_to_check_.emplace(0);
    }

    const auto& events = current_trigger.events;
    for (; *event_to_check_ < events.size(); ++*event_to_check_) {
      if (not events[*event_to_check_]->is_ready(box, cache, array_index,
                                                 component)) {
        return TriggeringState::NotReady;
      }
    }
    event_to_check_.reset();
  }

  for (size_t trigger_for_events_to_run = to_run_position_;
       trigger_for_events_to_run != events_and_triggers_.size();
       ++trigger_for_events_to_run) {
    for (const auto& event :
         events_and_triggers_[trigger_for_events_to_run].events) {
      if (event->needs_evolved_variables()) {
        return TriggeringState::NeedsEvolvedVariables;
      }
    }
  }

  return TriggeringState::Ready;
}

template <typename DbTags, typename Metavariables, typename ArrayIndex,
          typename ComponentPointer>
void EventsAndDenseTriggers::run_events(
    db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index, const ComponentPointer component) {
  ASSERT(initialized(), "Not initialized");
  ASSERT(not events_and_triggers_.empty(),
         "Should not be calling run_events with no triggers");
  using compute_tags = tmpl::remove_duplicates<tmpl::filter<
      tmpl::flatten<tmpl::transform<
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   Event>,
          get_tags<tmpl::_1>>>,
      db::is_compute_tag<tmpl::_1>>>;

  for (; to_run_position_ != events_and_triggers_.size(); ++to_run_position_) {
    const auto& trigger_to_run = events_and_triggers_[to_run_position_];
    db::mutate<::evolution::Tags::PreviousTriggerTime>(
        make_not_null(&box),
        [&trigger_to_run](
            const gsl::not_null<std::optional<double>*> previous_trigger_time) {
          *previous_trigger_time =
              trigger_to_run.trigger->previous_trigger_time();
        });
    const auto observation_box = make_observation_box<compute_tags>(box);
    for (const auto& event : trigger_to_run.events) {
      event->run(observation_box, cache, array_index, component);
    }
    db::mutate<::evolution::Tags::PreviousTriggerTime>(
        make_not_null(&box),
        [](const gsl::not_null<std::optional<double>*> previous_trigger_time) {
          *previous_trigger_time = std::numeric_limits<double>::signaling_NaN();
        });
  }
}

template <typename DbTags, typename Metavariables, typename ArrayIndex,
          typename ComponentPointer>
bool EventsAndDenseTriggers::reschedule(
    const db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index, const ComponentPointer component) {
  ASSERT(initialized(), "Not initialized");
  ASSERT(not events_and_triggers_.empty(),
         "Should not be calling run_events with no triggers");

  while (heap_end() != events_and_triggers_.end()) {
    const std::optional<double> next_check =
        heap_end()->trigger->next_check_time(box, cache, array_index,
                                             component);
    if (not next_check.has_value()) {
      return false;
    }
    reschedule_next_trigger(
        *next_check, db::get<::Tags::TimeStepId>(box).time_runs_forward());
  }

  populate_active_triggers();
  return true;
}

template <typename F>
void EventsAndDenseTriggers::for_each_event(F&& f) const {
  for (const auto& trigger_and_events : events_and_triggers_) {
    for (const auto& event : trigger_and_events.events) {
      f(*event);
    }
  }
}

template <typename DbTags>
void EventsAndDenseTriggers::initialize(const db::DataBox<DbTags>& box) {
  next_check_after_ =
      TriggerTimeAfter{db::get<::Tags::TimeStepId>(box).time_runs_forward()};
  next_check_ = db::get<::Tags::Time>(box);
  for (auto& trigger_record : events_and_triggers_) {
    trigger_record.next_check = next_check_;
  }
  heap_size_ = 0;
}
}  // namespace evolution

template <>
struct Options::create_from_yaml<evolution::EventsAndDenseTriggers> {
  using type = evolution::EventsAndDenseTriggers;
  template <typename Metavariables>
  static type create(const Options::Option& options) {
    return type(options.parse_as<typename type::ConstructionType,
                                 Metavariables>());
  }
};
