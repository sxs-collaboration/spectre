# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Burgers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Fluxes.cpp
  TimeDerivativeTerms.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Fluxes.hpp
  System.hpp
  Tags.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DgSubcell
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  ErrorHandling
  Events
  Evolution
  FiniteDifference
  Options
  Parallel
  Serialization
  Utilities
  PRIVATE
  BurgersAnalyticData
  BurgersSolutions
  LinearOperators
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Instantiations)
add_subdirectory(Subcell)
