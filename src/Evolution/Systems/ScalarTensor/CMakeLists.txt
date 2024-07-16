# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarTensor)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  StressEnergy.cpp
  TimeDerivative.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Constraints.hpp
  Initialize.hpp
  StressEnergy.hpp
  System.hpp
  Tags.hpp
  TimeDerivative.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  DomainBoundaryConditions
  GeneralRelativity
  GeneralizedHarmonic
  Serialization
  Utilities
  CurvedScalarWave
  INTERFACE
  Options
  Parallel
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(Sources)
