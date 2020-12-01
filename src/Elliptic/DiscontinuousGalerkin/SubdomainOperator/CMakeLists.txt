# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EllipticDgSubdomainOperator)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeSubdomain.hpp
  SubdomainOperator.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainStructure
  Elliptic
  EllipticDg
  ErrorHandling
  Initialization
  LinearOperators
  ParallelSchwarz
  Spectral
  Utilities
)
