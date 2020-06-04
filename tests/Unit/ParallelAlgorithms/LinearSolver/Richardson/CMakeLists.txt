# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelRichardson")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/Richardson"
  "${LIBRARY_SOURCES}"
  "ParallelLinearSolver"
  )

add_linear_solver_algorithm_test("RichardsonAlgorithm")
add_distributed_linear_solver_algorithm_test("DistributedRichardsonAlgorithm")
