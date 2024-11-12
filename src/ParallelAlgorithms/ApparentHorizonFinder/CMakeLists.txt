# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizonFinder)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FastFlow.cpp
  InterpolationTarget.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApparentHorizon.hpp
  ComputeExcisionBoundaryVolumeQuantities.hpp
  ComputeExcisionBoundaryVolumeQuantities.tpp
  ComputeHorizonVolumeQuantities.hpp
  ComputeHorizonVolumeQuantities.tpp
  FastFlow.hpp
  HorizonAliases.hpp
  InterpolationTarget.hpp
  ObserveCenters.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GrSurfaces
  FiniteDifference
  GeneralRelativity
  LinearAlgebra
  Logging
  Options
  Printf
  SphericalHarmonics
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
add_subdirectory(Python)
