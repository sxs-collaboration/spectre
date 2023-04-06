# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EllipticSubdomainPreconditioners)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RegisterDerived.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  MinusLaplacian.hpp
  RegisterDerived.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  LinearSolver
  Parallel
  ParallelSchwarz
  Poisson
  Serialization
  Utilities
  INTERFACE
  Convergence
  DataStructures
  Domain
  DomainStructure
  Elliptic
  EllipticDgSubdomainOperator
  ErrorHandling
  Logging
  Options
  PoissonBoundaryConditions
  )
