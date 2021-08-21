# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PhaseControl)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CheckpointAndExitAfterWallclock.hpp
  ExecutePhaseChange.hpp
  PhaseChange.hpp
  PhaseControlTags.hpp
  PhaseSelection.hpp
  VisitAndReturn.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  EventsAndTriggers
  Options
  Parallel
  Utilities
  )
