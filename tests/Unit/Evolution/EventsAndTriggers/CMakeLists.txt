# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EVENTS_AND_TRIGGERS_TESTS
    Evolution/EventsAndTriggers/Test_EventsAndTriggers.cpp
   )

set(EVOLUTION_TESTS "${EVOLUTION_TESTS};${EVENTS_AND_TRIGGERS_TESTS}"
    PARENT_SCOPE)
