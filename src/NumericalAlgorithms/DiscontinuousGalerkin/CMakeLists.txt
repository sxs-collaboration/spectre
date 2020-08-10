# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DiscontinuousGalerkin)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Formulation.cpp
  MortarHelpers.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Formulation.hpp
  LiftFlux.hpp
  MortarHelpers.hpp
  NormalDotFlux.hpp
  Protocols.hpp
  SimpleBoundaryData.hpp
  SimpleMortarData.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  DomainStructure
  Options
  Spectral
  INTERFACE
  Domain
  )

add_subdirectory(Actions)
add_subdirectory(BoundarySchemes)
add_subdirectory(NumericalFluxes)
