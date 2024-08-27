# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelGmres")

set(LIBRARY_SOURCES
  Test_ElementActions.cpp
  Test_ResidualMonitorActions.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Convergence
  DataStructures
  LinearSolverHelpers
  Logging
  Observer
  ParallelLinearSolver
)

add_standalone_test(
  "Integration.LinearSolver.GmresAlgorithm"
  INPUT_FILE "Test_GmresAlgorithm.yaml")
target_link_libraries(
  "Test_GmresAlgorithm"
  PRIVATE
  "${INTEGRATION_TEST_LINK_LIBRARIES}")
add_standalone_test(
  "Integration.LinearSolver.ComplexGmresAlgorithm"
  INPUT_FILE "Test_ComplexGmresAlgorithm.yaml")
target_link_libraries(
  "Test_ComplexGmresAlgorithm"
  PRIVATE
  "${INTEGRATION_TEST_LINK_LIBRARIES}")
add_standalone_test(
  "Integration.LinearSolver.GmresPreconditionedAlgorithm"
  INPUT_FILE "Test_GmresPreconditionedAlgorithm.yaml")
target_link_libraries(
  "Test_GmresPreconditionedAlgorithm"
  PRIVATE
  "${INTEGRATION_TEST_LINK_LIBRARIES}")
add_standalone_test(
  "Integration.LinearSolver.DistributedGmresAlgorithm"
  INPUT_FILE "Test_DistributedGmresAlgorithm.yaml")
target_link_libraries(
  "Test_DistributedGmresAlgorithm"
  PRIVATE
  "${DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES}")
add_standalone_test(
  "Integration.LinearSolver.DistributedGmresPreconditionedAlgorithm"
  INPUT_FILE "Test_DistributedGmresPreconditionedAlgorithm.yaml")
target_link_libraries(
  "Test_DistributedGmresPreconditionedAlgorithm"
  PRIVATE
  "${DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES}")
