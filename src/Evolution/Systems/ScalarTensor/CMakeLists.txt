# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarTensor)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  TimeDerivative.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Constraints.hpp
  Initialize.hpp
  System.hpp
  Tags.hpp
  TimeDerivative.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ConstraintDamping
  CurvedScalarWave
  DataStructures
  DiscontinuousGalerkin
  DomainBoundaryConditions
  GeneralRelativity
  GeneralizedHarmonic
  GhScalarTensorAnalyticData
  ScalarTensorPointwise
  Serialization
  Utilities
  INTERFACE
  Options
  Parallel
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(Instantiations)
