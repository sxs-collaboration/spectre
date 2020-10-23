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
  "ParallelNonlinearSolver"
  )
