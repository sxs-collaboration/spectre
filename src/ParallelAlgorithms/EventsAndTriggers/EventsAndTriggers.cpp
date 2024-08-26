// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"

#include <algorithm>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

EventsAndTriggers::TriggerAndEvents::TriggerAndEvents() = default;

EventsAndTriggers::TriggerAndEvents::TriggerAndEvents(
    std::unique_ptr<::Trigger> trigger_in,
    std::vector<std::unique_ptr<::Event>> events_in)
    : trigger(std::move(trigger_in)), events(std::move(events_in)) {}

void EventsAndTriggers::TriggerAndEvents::pup(PUP::er& p) {
  p | trigger;
  p | events;
}

EventsAndTriggers::EventsAndTriggers() = default;

EventsAndTriggers::EventsAndTriggers(Storage events_and_triggers)
    : events_and_triggers_(std::move(events_and_triggers)) {}

void EventsAndTriggers::pup(PUP::er& p) { p | events_and_triggers_; }
