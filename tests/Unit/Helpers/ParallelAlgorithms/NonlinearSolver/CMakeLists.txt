# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "NonlinearSolverHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  Algorithm.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Actions
  Convergence
  DataStructures
  ErrorHandling
  Initialization
  LinearSolverHelpers
  Observer
  Options
  Parallel
  ParallelLinearSolver
  ParallelNonlinearSolver
  Utilities
  )
