# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelConjugateGradient")

set(LIBRARY_SOURCES
  Test_ElementActions.cpp
  Test_ResidualMonitorActions.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/ConjugateGradient"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Convergence
  DataStructures
  IO
  Logging
  ParallelLinearSolver
)

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

add_linear_solver_algorithm_test("ConjugateGradientAlgorithm")
add_distributed_linear_solver_algorithm_test(
  "DistributedConjugateGradientAlgorithm")
