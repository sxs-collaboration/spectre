# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PhaseControl)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CheckpointAndExitAfterWallclock.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CheckpointAndExitAfterWallclock.hpp
  ContributeToPhaseChangeReduction.hpp
  ExecutePhaseChange.hpp
  Factory.hpp
  InitializePhaseChangeDecisionData.hpp
  PhaseChange.hpp
  PhaseControlTags.hpp
  VisitAndReturn.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  EventsAndTriggers
  PUBLIC
  Charmxx::charmxx
  Options
  Parallel
  Serialization
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_Main
  )
