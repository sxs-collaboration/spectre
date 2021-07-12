# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DiscontinuousGalerkin)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyMassMatrix.cpp
  Formulation.cpp
  MetricIdentityJacobian.cpp
  MortarHelpers.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyMassMatrix.hpp
  Formulation.hpp
  HasReceivedFromAllMortars.hpp
  LiftFlux.hpp
  MetricIdentityJacobian.hpp
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
  ErrorHandling
  Options
  Spectral
  Utilities
  INTERFACE
  Domain
  )

add_subdirectory(NumericalFluxes)
add_subdirectory(Tags)
