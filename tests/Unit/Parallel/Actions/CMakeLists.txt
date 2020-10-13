# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelActions")

set(LIBRARY_SOURCES
  Test_Goto.cpp
  Test_SetupDataBox.cpp
  Test_TerminatePhase.cpp
  )

add_test_library(
  ${LIBRARY}
  "Parallel/Actions"
  "${LIBRARY_SOURCES}"
  ""
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Parallel
  )
