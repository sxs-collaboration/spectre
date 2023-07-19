# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Initialization")

set(LIBRARY_SOURCES
  Test_MutateAssign.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Initialization/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  Utilities
  )
