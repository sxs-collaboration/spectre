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
  "Boost::boost;DataStructures;DiscontinuousGalerkin;Domain;ErrorHandling;EventsAndDenseTriggers"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
