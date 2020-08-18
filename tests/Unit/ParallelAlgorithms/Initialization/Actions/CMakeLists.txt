# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_InitializationActions")

set(LIBRARY_SOURCES
  Test_AddComputeTags.cpp
  Test_RemoveOptionsAndTerminatePhase.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Initialization/Actions"
  "${LIBRARY_SOURCES}"
  "DataStructures;ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
