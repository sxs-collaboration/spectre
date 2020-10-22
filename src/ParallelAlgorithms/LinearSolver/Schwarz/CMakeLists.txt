# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelSchwarz)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  OverlapHelpers.cpp
  Weighting.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeTags.hpp
  ElementActions.hpp
  ElementCenteredSubdomainData.hpp
  OverlapHelpers.hpp
  Protocols.hpp
  Schwarz.hpp
  Tags.hpp
  Weighting.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DomainStructure
  ErrorHandling
  Utilities
  INTERFACE
  Boost::boost
  Convergence
  Domain
  IO
  Informer
  Initialization
  LinearSolver
  Options
  Parallel
  ParallelDg
  ParallelLinearSolver
  Spectral
  )

add_subdirectory(Actions)
