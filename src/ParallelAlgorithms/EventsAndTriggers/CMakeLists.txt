# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EventsAndTriggers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Completion.cpp
  LogicalTriggers.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Completion.hpp
  Event.hpp
  EventsAndTriggers.hpp
  LogicalTriggers.hpp
  Tags.hpp
  Trigger.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  ErrorHandling
  PUBLIC
  Options
  Parallel
  Utilities
  )

add_subdirectory(Actions)
