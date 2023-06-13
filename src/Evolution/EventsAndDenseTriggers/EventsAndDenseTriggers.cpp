// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"

#include <algorithm>
#include <pup.h>
#include <utility>

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

namespace evolution {
void EventsAndDenseTriggers::TriggerRecord::pup(PUP::er& p) {
  p | next_check;
  p | is_triggered;
  p | num_events_ready;
  p | trigger;
  p | events;
}

EventsAndDenseTriggers::EventsAndDenseTriggers(
    ConstructionType events_and_triggers) {
  events_and_triggers_.reserve(events_and_triggers.size());
  for (auto& events_and_trigger : events_and_triggers) {
    events_and_triggers_.push_back(TriggerRecord{
        std::numeric_limits<double>::signaling_NaN(), std::optional<bool>{}, 0,
        std::move(events_and_trigger.trigger),
        std::move(events_and_trigger.events)});
  }
}

void EventsAndDenseTriggers::add_trigger_and_events(
    std::unique_ptr<DenseTrigger> trigger,
    std::vector<std::unique_ptr<Event>> events) {
  ASSERT(not initialized(), "Cannot add events after initialization");
  events_and_triggers_.reserve(events_and_triggers_.size() + 1);
  events_and_triggers_.push_back(TriggerRecord{
      std::numeric_limits<double>::signaling_NaN(), std::optional<bool>{}, 0,
      std::move(trigger), std::move(events)});
}

void EventsAndDenseTriggers::pup(PUP::er& p) {
  p | events_and_triggers_;
  p | next_check_;
  p | before_;
}

bool EventsAndDenseTriggers::initialized() const {
  const ScopedFpeState disable_fpes(false);
  return not std::isnan(next_check_);
}
}  // namespace evolution
