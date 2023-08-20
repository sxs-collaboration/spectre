# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionActions")

set(LIBRARY_SOURCES
  Test_RunEventsAndDenseTriggers.cpp
  Test_RunEventsAndTriggers.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  DiscontinuousGalerkin
  Domain
  ErrorHandling
  EventsAndDenseTriggers
  Parallel
  Time
  Utilities
  )

