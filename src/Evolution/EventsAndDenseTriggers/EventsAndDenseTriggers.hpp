// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
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
    void pup(PUP::er& p) noexcept;
  };
  using Storage = std::vector<TriggerRecord>;

 public:
  EventsAndDenseTriggers() = default;
  explicit EventsAndDenseTriggers(
      ConstructionType events_and_triggers) noexcept;

  template <typename DbTags>
  double next_trigger(const db::DataBox<DbTags>& box) noexcept;

  enum class TriggeringState { Ready, NeedsEvolvedVariables, NotReady };

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  TriggeringState is_ready(const db::DataBox<DbTags>& box,
                           Parallel::GlobalCache<Metavariables>& cache,
                           const ArrayIndex& array_index,
                           const ComponentPointer component) noexcept;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  void run_events(db::DataBox<DbTags>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ComponentPointer component) noexcept;

  /// Add a new trigger and set of events.  This can only be called
  /// during initialization.
  void add_trigger_and_events(
      std::unique_ptr<DenseTrigger> trigger,
      std::vector<std::unique_ptr<Event>> events) noexcept;

  template <typename F>
  void for_each_event(F&& f) const noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  typename Storage::iterator heap_end() noexcept {
    return events_and_triggers_.begin() + heap_size_;
  }

  typename Storage::iterator current_trigger() noexcept {
    return events_and_triggers_.begin() + processing_position_;
  }

  template <typename DbTags>
  void initialize(const db::DataBox<DbTags>& box) noexcept;

  void populate_active_triggers() noexcept;

  void finish_processing_trigger_at_current_time(
      const typename Storage::iterator& index) noexcept;

  class TriggerTimeAfter {
   public:
    explicit TriggerTimeAfter(const bool time_runs_forward);

    bool operator()(const TriggerRecord& a,
                    const TriggerRecord& b) const noexcept;

    double infinite_future() const noexcept;

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p) noexcept;

   private:
    evolution_greater<double> time_after_{};
  };

  // The data structure used here is a heap containing the triggers
  // not being tested at the moment, with the earliest time as the
  // root at index 0, followed by the triggers known to trigger at the
  // current time, followed by the triggers currently being processed.
  // When we are done with the current time, we choose the next time
  // to process by examining the heap root and then pop all triggers
  // requesting that time into the processing area.  Triggers are
  // moved back to the heap either when we know they don't trigger or
  // when we have run their events.
  //
  // Except at initialization (when everything needs to be processed
  // at the start time), the trigger processing area is expected to be
  // small.
  Storage events_and_triggers_;
  // The size of the heap, or equivalently the index of the start of
  // the processing area.  The initial -1 is a sentinel for an
  // uninitialized state.
  typename Storage::difference_type heap_size_ = -1;
  // Index of the next trigger to process.
  typename Storage::difference_type processing_position_{};
  double next_check_ = std::numeric_limits<double>::signaling_NaN();
  TriggerTimeAfter next_check_after_{false};
};

template <typename DbTags>
double EventsAndDenseTriggers::next_trigger(
    const db::DataBox<DbTags>& box) noexcept {
  if (UNLIKELY(heap_size_ == -1)) {
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
    const ArrayIndex& array_index, const ComponentPointer component) noexcept {
  ASSERT(heap_size_ != -1, "Not initialized");
  ASSERT(not events_and_triggers_.empty(),
         "Should not be calling is_ready with no triggers");
  ASSERT(heap_end() != events_and_triggers_.end(), "No active triggers");

  const evolution_greater<double> after{
      db::get<::Tags::TimeStepId>(box).time_runs_forward()};

  while (current_trigger() != events_and_triggers_.end()) {
    if (not current_trigger()->trigger->is_ready(box, cache, array_index,
                                                 component)) {
      return TriggeringState::NotReady;
    }

    const auto is_triggered = current_trigger()->trigger->is_triggered(box);
    if (not after(is_triggered.next_check, current_trigger()->next_check)) {
      ERROR("Trigger at time " << current_trigger()->next_check
            << " rescheduled itself for earlier time "
            << is_triggered.next_check);
    }
    current_trigger()->next_check = is_triggered.next_check;
    if (not is_triggered.is_triggered) {
      finish_processing_trigger_at_current_time(current_trigger());
    }
    ++processing_position_;
  }

  // The triggers we are processing are not in the heap, but are
  // stored after it in the data array.
  for (auto trigger = heap_end();
       trigger != events_and_triggers_.end();
       ++trigger) {
    for (const auto& event : trigger->events) {
      if (not event->is_ready(box, cache, array_index, component)) {
        return TriggeringState::NotReady;
      }
    }
  }

  for (auto trigger = heap_end();
       trigger != events_and_triggers_.end();
       ++trigger) {
    for (const auto& event : trigger->events) {
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
    const ArrayIndex& array_index, const ComponentPointer component) noexcept {
  ASSERT(heap_size_ != -1, "Not initialized");
  ASSERT(not events_and_triggers_.empty(),
         "Should not be calling run_events with no triggers");

  for (auto trigger = heap_end();
       trigger != events_and_triggers_.end();
       ++trigger) {
    db::mutate<::evolution::Tags::PreviousTriggerTime>(
        make_not_null(&box),
        [&trigger](const gsl::not_null<std::optional<double>*>
                       previous_trigger_time) noexcept {
          *previous_trigger_time = trigger->trigger->previous_trigger_time();
        });
    for (const auto& event : trigger->events) {
      event->run(box, cache, array_index, component);
    }
    db::mutate<::evolution::Tags::PreviousTriggerTime>(
        make_not_null(&box), [](const gsl::not_null<std::optional<double>*>
                                    previous_trigger_time) noexcept {
          *previous_trigger_time = std::numeric_limits<double>::signaling_NaN();
        });
    finish_processing_trigger_at_current_time(trigger);
  }

  populate_active_triggers();
}

template <typename F>
void EventsAndDenseTriggers::for_each_event(F&& f) const noexcept {
  for (const auto& trigger_and_events : events_and_triggers_) {
    for (const auto& event : trigger_and_events.events) {
      f(*event);
    }
  }
}

template <typename DbTags>
void EventsAndDenseTriggers::initialize(
    const db::DataBox<DbTags>& box) noexcept {
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
