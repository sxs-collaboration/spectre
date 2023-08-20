# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PhaseControl")

set(LIBRARY_SOURCES
  Test_CheckpointAndExitAfterWallclock.cpp
  Test_ExecutePhaseChange.cpp
  Test_PhaseChange.cpp
  Test_PhaseControlTags.cpp
  Test_VisitAndReturn.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}" WITH_CHARM)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Options
  Parallel
  PhaseControl
  Time
  Utilities
  )
