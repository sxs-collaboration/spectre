# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelMultigrid)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Hierarchy.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Hierarchy.hpp
  Multigrid.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DomainStructure
  ErrorHandling
  Utilities
  INTERFACE
  Convergence
  DataStructures
  Domain
  Initialization
  IO
  Logging
  Options
  Parallel
  ParallelLinearSolver
  Spectral
  SystemUtilities
  )
