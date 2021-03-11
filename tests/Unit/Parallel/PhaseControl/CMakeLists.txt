# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PhaseControl")

set(LIBRARY_SOURCES
  Test_ExecutePhaseChange.cpp
  Test_PhaseChange.cpp
  Test_PhaseControlTags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Parallel/PhaseControl"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Parallel
  )
