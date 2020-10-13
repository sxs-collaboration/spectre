# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Initialization")

set(LIBRARY_SOURCES
  Test_MergeIntoDataBox.cpp
  Test_MutateAssign.cpp
  )

add_subdirectory(Actions)

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Initialization/"
  "${LIBRARY_SOURCES}"
  "DataStructures;ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
