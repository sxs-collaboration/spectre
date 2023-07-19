# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionTriggers")

set(LIBRARY_SOURCES
  Test_SeparationLessThan.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Triggers/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  EventsAndTriggers
  EvolutionTriggers
  FunctionsOfTime
  Options
  Utilities
)
