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
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ParallelLinearSolver
  )

add_standalone_test(
  "Integration.LinearSolver.RichardsonAlgorithm"
  INPUT_FILE "Test_RichardsonAlgorithm.yaml")
target_link_libraries(
  "Test_RichardsonAlgorithm"
  PRIVATE
  "${INTEGRATION_TEST_LINK_LIBRARIES}")
add_standalone_test(
  "Integration.LinearSolver.DistributedRichardsonAlgorithm"
  INPUT_FILE "Test_DistributedRichardsonAlgorithm.yaml")
target_link_libraries(
  "Test_DistributedRichardsonAlgorithm"
  PRIVATE
  "${DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES}")
