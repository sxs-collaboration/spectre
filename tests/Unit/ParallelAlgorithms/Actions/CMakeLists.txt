# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelAlgorithmsActions")

set(LIBRARY_SOURCES
  Test_MutateApply.cpp
  Test_SetData.cpp
  Test_RandomizeVariables.cpp
  Test_UpdateMessageQueue.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Actions/"
  "${LIBRARY_SOURCES}"
  "DataStructures"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
