# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EvolutionTriggers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  SeparationLessThan.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  SeparationLessThan.hpp
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
