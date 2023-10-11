# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_LinearSolverActions")

set(LIBRARY_SOURCES
  Test_MakeIdentityIfSkipped.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Convergence
  DataStructures
  Parallel
  Utilities
  )

add_standalone_test(
  "Integration.LinearSolver.BuildMatrix"
  INPUT_FILE "Test_BuildMatrix.yaml")
target_link_libraries(
  Test_BuildMatrix
  PRIVATE
  Charmxx::main
  Convergence
  DomainCreators
  DomainStructure
  H5
  Options
  Parallel
  ParallelLinearSolver
  Utilities
  )
