# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ForceFree)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  CoordinateMaps
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  DomainStructure
  DgSubcell
  ErrorHandling
  Evolution
  FiniteDifference
  GeneralRelativity
  Options
  Parallel
  Spectral
  Utilities
  )
