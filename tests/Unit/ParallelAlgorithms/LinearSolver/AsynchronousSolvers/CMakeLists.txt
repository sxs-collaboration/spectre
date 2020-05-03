# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_AsynchronousLinearSolvers")

set(LIBRARY_SOURCES
  Test_ElementActions.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/AsynchronousSolvers"
  "${LIBRARY_SOURCES}"
  "DataStructures;IO;ParallelLinearSolver"
  )
