# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarWave)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Constraints.cpp
  EnergyDensity.cpp
  Equations.cpp
  TimeDerivative.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Constraints.hpp
  EnergyDensity.hpp
  Equations.hpp
  Initialize.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  ErrorHandling
  LinearOperators
  Options
  Utilities
  INTERFACE
  Initialization
  Parallel
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
