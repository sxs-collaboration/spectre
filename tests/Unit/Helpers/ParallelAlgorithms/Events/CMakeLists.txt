# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "EventsHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  ObserveFields.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  AnalyticSolutions
  DataStructures
  Domain
  DomainStructure
  Observer
  Options
  Parallel
  Utilities
  )
