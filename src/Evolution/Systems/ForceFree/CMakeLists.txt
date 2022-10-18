# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ForceFree)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Fluxes.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Fluxes.hpp
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

add_subdirectory(BoundaryConditions)
