# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EvolutionTriggers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FractionOfOrbit.cpp
  SeparationLessThan.cpp
  NumberOfOrbits.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FractionOfOrbit.hpp
  SeparationLessThan.hpp
  NumberOfOrbits.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  EventsAndTriggers
  FunctionsOfTime
  Options
  Utilities
  )
