# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizons)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FastFlow.cpp
  TimeDerivStrahlkorper.cpp
  StrahlkorperGr.cpp
  StrahlkorperCoordsInDifferentFrame.cpp
  StrahlkorperInDifferentFrame.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeExcisionBoundaryVolumeQuantities.hpp
  ComputeExcisionBoundaryVolumeQuantities.tpp
  ComputeHorizonVolumeQuantities.hpp
  ComputeHorizonVolumeQuantities.tpp
  ComputeItems.hpp
  FastFlow.hpp
  HorizonAliases.hpp
  ObserveCenters.hpp
  StrahlkorperGr.hpp
  StrahlkorperCoordsInDifferentFrame.hpp
  StrahlkorperInDifferentFrame.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivStrahlkorper.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  FiniteDifference
  GeneralRelativity
  LinearAlgebra
  Options
  SphericalHarmonics
  PRIVATE
  CoordinateMaps
  Domain
  INTERFACE
  Observer
  )
