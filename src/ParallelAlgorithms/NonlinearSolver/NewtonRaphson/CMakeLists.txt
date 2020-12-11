# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ParallelNonlinearSolver
  PRIVATE
  LineSearch.cpp
  )

spectre_target_headers(
  ParallelNonlinearSolver
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementActions.hpp
  LineSearch.hpp
  NewtonRaphson.hpp
  ResidualMonitor.hpp
  ResidualMonitorActions.hpp
  )

add_subdirectory(Tags)
