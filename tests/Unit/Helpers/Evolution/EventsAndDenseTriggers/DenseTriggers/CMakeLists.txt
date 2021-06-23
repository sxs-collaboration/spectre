# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DenseTriggerHelpers")

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  TestTrigger.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  TestTrigger.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  EventsAndDenseTriggers
  Options
  Parallel
  Utilities
  )
