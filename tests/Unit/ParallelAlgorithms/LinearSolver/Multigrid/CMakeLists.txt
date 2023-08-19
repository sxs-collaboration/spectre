# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelMultigrid")

set(LIBRARY_SOURCES
  Test_Hierarchy.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainStructure
  ParallelMultigrid
  Utilities
  )

add_standalone_test(
  "Integration.LinearSolver.MultigridAlgorithm"
  INPUT_FILE "Test_MultigridAlgorithm.yaml")
add_standalone_test(
  "Integration.LinearSolver.MultigridAlgorithmMassive"
  EXECUTABLE "Test_MultigridAlgorithm"
  INPUT_FILE "Test_MultigridAlgorithmMassive.yaml")
target_link_libraries(
  "Test_MultigridAlgorithm"
  PRIVATE
  "${DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES};ParallelMultigrid")

add_standalone_test(
  "Integration.LinearSolver.MultigridPreconditionedGmresAlgorithm"
  INPUT_FILE "Test_MultigridPreconditionedGmresAlgorithm.yaml")
target_link_libraries(
  "Test_MultigridPreconditionedGmresAlgorithm"
  PRIVATE
  "${DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES};ParallelMultigrid;ParallelNonlinearSolver")

add_subdirectory(Actions)
