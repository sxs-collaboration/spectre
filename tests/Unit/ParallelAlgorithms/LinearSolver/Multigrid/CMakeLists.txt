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

add_distributed_linear_solver_algorithm_test("MultigridAlgorithm")
target_link_libraries(
  Test_MultigridAlgorithm
  PRIVATE
  ParallelMultigrid
  )

add_test(
  NAME Integration.LinearSolver.MultigridAlgorithmMassive
  COMMAND ${CMAKE_BINARY_DIR}/bin/Test_MultigridAlgorithm --input-file
  ${CMAKE_CURRENT_SOURCE_DIR}/Test_MultigridAlgorithmMassive.yaml
  )
set_tests_properties(
  Integration.LinearSolver.MultigridAlgorithmMassive
  PROPERTIES
  TIMEOUT 5
  LABELS "integration"
  ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

add_distributed_linear_solver_algorithm_test(
  "MultigridPreconditionedGmresAlgorithm")
target_link_libraries(
  Test_MultigridPreconditionedGmresAlgorithm
  PRIVATE
  ParallelMultigrid
  )

add_subdirectory(Actions)
