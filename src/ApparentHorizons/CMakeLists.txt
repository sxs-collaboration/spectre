# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizons)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FastFlow.cpp
  ObjectLabel.cpp
  StrahlkorperGr.cpp
  StrahlkorperInDifferentFrame.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeHorizonVolumeQuantities.hpp
  ComputeHorizonVolumeQuantities.tpp
  ComputeItems.hpp
  FastFlow.hpp
  ObjectLabel.hpp
  ObserveCenters.hpp
  StrahlkorperGr.hpp
  StrahlkorperInDifferentFrame.hpp
  Tags.hpp
  TagsDeclarations.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralRelativity
  LinearAlgebra
  Options
  SphericalHarmonics
  PRIVATE
  CoordinateMaps
  Domain
  )
