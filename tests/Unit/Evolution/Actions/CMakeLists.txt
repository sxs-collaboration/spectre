# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionActions")

set(LIBRARY_SOURCES
  Test_RunEventsAndDenseTriggers.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Actions/"
  "${LIBRARY_SOURCES}"
  ""
  )

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

