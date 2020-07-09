# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EventsAndTriggers)

add_spectre_library(${LIBRARY} INTERFACE)

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
  Domain
  ErrorHandling
  Utilities
  )

add_subdirectory(Actions)
