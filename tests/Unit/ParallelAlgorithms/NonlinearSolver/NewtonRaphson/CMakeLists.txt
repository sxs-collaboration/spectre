# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelNewtonRaphson")

set(LIBRARY_SOURCES
  Test_LineSearch.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/NonlinearSolver/NewtonRaphson"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ParallelNonlinearSolver
  )

set(INTEGRATION_TEST_LINK_LIBRARIES
  # Link against Boost::program_options for now until we have proper
  # dependency handling for header-only libs
  Boost::program_options
  Charmxx::main
  Convergence
  DataStructures
  ErrorHandling
  Informer
  ParallelLinearSolver
  ParallelNonlinearSolver
  )

add_standalone_test(
  "Integration.LinearSolver.NewtonRaphsonAlgorithm"
  INPUT_FILE "Test_NewtonRaphsonAlgorithm.yaml")
target_link_libraries(
  "Test_NewtonRaphsonAlgorithm"
  PRIVATE
  "${INTEGRATION_TEST_LINK_LIBRARIES}")
