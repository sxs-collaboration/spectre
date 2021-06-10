# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelMultigrid")

set(LIBRARY_SOURCES
  Test_Hierarchy.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/Multigrid"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainStructure
  ParallelMultigrid
  Utilities
  )
