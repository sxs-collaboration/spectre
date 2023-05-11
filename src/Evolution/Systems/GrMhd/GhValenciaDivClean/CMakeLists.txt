# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GhValenciaDivClean)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  StressEnergy.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Constraints.hpp
  SetPiFromGauge.hpp
  System.hpp
  StressEnergy.hpp
  Tags.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  DomainBoundaryConditions
  FiniteDifference
  GeneralRelativity
  GeneralizedHarmonic
  Hydro
  Serialization
  Utilities
  ValenciaDivClean
  INTERFACE
  Options
  Parallel
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Subcell)
