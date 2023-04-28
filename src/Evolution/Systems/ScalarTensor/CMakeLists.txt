# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarTensor)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  StressEnergy.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  StressEnergy.hpp
  System.hpp
  Tags.hpp
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