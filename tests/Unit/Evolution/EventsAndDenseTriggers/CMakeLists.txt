# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EventsAndDenseTriggers")

set(LIBRARY_SOURCES
  Test_EventsAndDenseTriggers.cpp
  )

add_subdirectory(DenseTriggers)

add_test_library(
  ${LIBRARY}
  "Evolution/EventsAndDenseTriggers/"
  "${LIBRARY_SOURCES}"
  "DataStructures;DenseTriggerHelpers;EventsAndDenseTriggers;EventsAndTriggers;Options;Parallel;Time;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
