# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizonFinder)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CleanUp.cpp
  ComputeCoords.cpp
  ComputeVarsToInterpolateToTarget.cpp
  CurrentTime.cpp
  Destination.cpp
  FastFlow.cpp
  InterpolateVolumeVars.cpp
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
  ComputeCoords.hpp
  ComputeVarsToInterpolateToTarget.hpp
  CurrentTime.hpp
  Destination.hpp
  FastFlow.hpp
  FindApparentHorizon.hpp
  HorizonAliases.hpp
  Initialization.hpp
  InterpolateVolumeVars.hpp
  OptionTags.hpp
  PrintDeadlockAnalysis.hpp
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
  LinearOperators
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
  )

add_subdirectory(Callbacks)
add_subdirectory(Criteria)
add_subdirectory(Events)
add_subdirectory(Protocols)
add_subdirectory(Python)
