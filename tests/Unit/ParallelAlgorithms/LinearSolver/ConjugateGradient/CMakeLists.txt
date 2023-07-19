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
  Logging
  Observer
  ParallelLinearSolver
)

add_standalone_test(
  "Integration.LinearSolver.ConjugateGradientAlgorithm"
  INPUT_FILE "Test_ConjugateGradientAlgorithm.yaml")
target_link_libraries(
  "Test_ConjugateGradientAlgorithm"
  PRIVATE
  "${INTEGRATION_TEST_LINK_LIBRARIES}")
add_standalone_test(
  "Integration.LinearSolver.DistributedConjugateGradientAlgorithm"
  INPUT_FILE "Test_DistributedConjugateGradientAlgorithm.yaml")
target_link_libraries(
  "Test_DistributedConjugateGradientAlgorithm"
  PRIVATE
  "${DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES}")
