# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EventsAndDenseTriggers")

set(LIBRARY_SOURCES
  Test_EventsAndDenseTriggers.cpp
  Test_Tags.cpp
  )

add_subdirectory(DenseTriggers)

add_test_library(
  ${LIBRARY}
  "Evolution/EventsAndDenseTriggers/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DenseTriggerHelpers
  EventsAndDenseTriggers
  EventsAndTriggers
  Options
  Parallel
  Time
  Utilities
  )
