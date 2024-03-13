# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "LinearSolverHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  DistributedLinearSolverAlgorithmTestHelpers.hpp
  LinearSolverAlgorithmTestHelpers.hpp
  ResidualMonitorActionsTestHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Actions
  Convergence
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  EllipticDg
  ErrorHandling
  Initialization
  Observer
  Options
  Parallel
  ParallelLinearSolver
  Spectral
  Utilities
  )
