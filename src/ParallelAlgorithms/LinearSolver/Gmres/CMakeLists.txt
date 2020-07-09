# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ParallelLinearSolver
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementActions.hpp
  Gmres.hpp
  InitializeElement.hpp
  ResidualMonitor.hpp
  ResidualMonitorActions.hpp
  )

