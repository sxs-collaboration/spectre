# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_LinearSolverActions")

set(LIBRARY_SOURCES
  Test_TerminateIfConverged.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/LinearSolver/Actions"
  "${LIBRARY_SOURCES}"
  "Convergence;DataStructures"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
