# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EventsAndTriggers")

set(LIBRARY_SOURCES
  Test_EventsAndTriggers.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/EventsAndTriggers/"
  "${LIBRARY_SOURCES}"
  "ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Parallel
  )
