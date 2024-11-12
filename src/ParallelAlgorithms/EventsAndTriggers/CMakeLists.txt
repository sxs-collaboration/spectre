# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EventsAndTriggers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Completion.cpp
  EventsAndTriggers.cpp
  LogicalTriggers.cpp
  WhenToCheck.cpp
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
  WhenToCheck.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  ErrorHandling
  PUBLIC
  Options
  Parallel
  Serialization
  Utilities
  )

add_subdirectory(Actions)
