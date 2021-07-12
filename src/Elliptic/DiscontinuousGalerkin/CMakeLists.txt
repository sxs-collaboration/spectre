# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EllipticDg)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Initialization.cpp
  Penalty.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DgElementArray.hpp
  DgOperator.hpp
  Initialization.hpp
  Penalty.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainStructure
  ErrorHandling
  Spectral
  Utilities
  INTERFACE
  Boost::boost
  DomainCreators
  Elliptic
  EllipticSystems
  Initialization
  LinearOperators
  Options
  Parallel
  SystemUtilities
  )

add_subdirectory(Actions)
add_subdirectory(SubdomainOperator)
