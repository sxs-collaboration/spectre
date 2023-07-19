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
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Utilities
  ParallelNonlinearSolver
  )

add_subdirectory(NewtonRaphson)
