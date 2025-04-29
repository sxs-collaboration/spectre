# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizonFinder)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CleanUp.cpp
  ComputePoints.cpp
  ComputeVarsToInterpolateToTarget.cpp
  CurrentTime.cpp
  Destination.cpp
  FastFlow.cpp
  InterpolateVolumeVars.cpp
  InterpolationTarget.cpp
  OptionTags.cpp
  Storage.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApparentHorizon.hpp
  CleanUp.hpp
  Component.hpp
  ComputeExcisionBoundaryVolumeQuantities.hpp
  ComputeExcisionBoundaryVolumeQuantities.tpp
  ComputeHorizonVolumeQuantities.hpp
  ComputeHorizonVolumeQuantities.tpp
  ComputePoints.hpp
  ComputeVarsToInterpolateToTarget.hpp
  CurrentTime.hpp
  Destination.hpp
  FastFlow.hpp
  FindCommonHorizon.hpp
  HorizonAliases.hpp
  Initialization.hpp
  InterpolateVolumeVars.hpp
  InterpolationTarget.hpp
  OptionTags.hpp
  ReceiveVolumeData.hpp
  Storage.hpp
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
add_subdirectory(Events)
add_subdirectory(Python)
