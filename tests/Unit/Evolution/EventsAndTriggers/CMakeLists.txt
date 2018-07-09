# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionEventsAndTriggers")

set(LIBRARY_SOURCES
  Test_EventsAndTriggers.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/EventsAndTriggers/"
  "${LIBRARY_SOURCES}"
  "ErrorHandling;Utilities"
  )
