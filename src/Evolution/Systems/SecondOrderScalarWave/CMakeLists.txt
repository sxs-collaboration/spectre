# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SecondOrderScalarWave)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  TimeDerivative.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  UpdateAuxiliaryVariables.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  LinearOperators
  Utilities
  PRIVATE
  WaveEquationSolutions
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
