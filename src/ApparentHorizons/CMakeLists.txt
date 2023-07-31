# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizons)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FastFlow.cpp
  Target.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeExcisionBoundaryVolumeQuantities.hpp
  ComputeExcisionBoundaryVolumeQuantities.tpp
  ComputeHorizonVolumeQuantities.hpp
  ComputeHorizonVolumeQuantities.tpp
  FastFlow.hpp
  HorizonAliases.hpp
  ObserveCenters.hpp
  Tags.hpp
  Target.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  FiniteDifference
  GeneralRelativity
  LinearAlgebra
  Logging
  Options
  SphericalHarmonics
  StrahlkorperGr
  PRIVATE
  CoordinateMaps
  Domain
  INTERFACE
  Initialization
  Observer
  Parallel
  ParallelInterpolation
  )

  add_subdirectory(Callbacks)
