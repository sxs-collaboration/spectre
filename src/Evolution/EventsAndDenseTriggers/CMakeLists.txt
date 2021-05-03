# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EventsAndDenseTriggers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  EventsAndDenseTriggers.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DenseTrigger.hpp
  EventsAndDenseTriggers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  EventsAndTriggers
  Options
  Parallel
  Time
  Utilities
  )

add_subdirectory(DenseTriggers)
