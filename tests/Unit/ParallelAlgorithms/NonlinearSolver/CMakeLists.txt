# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelNonlinearSolver")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/NonlinearSolver/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities;ParallelNonlinearSolver"
  )
