# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Initialization")

set(LIBRARY_SOURCES
  Actions/Test_RemoveOptionsAndTerminatePhase.cpp
  Test_MergeIntoDataBox.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Initialization/"
  "${LIBRARY_SOURCES}"
  "DataStructures;ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
