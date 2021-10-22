# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizons)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ChangeCenterOfStrahlkorper.cpp
  FastFlow.cpp
  Strahlkorper.cpp
  StrahlkorperGr.cpp
  StrahlkorperInDifferentFrame.cpp
  Tags.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ChangeCenterOfStrahlkorper.hpp
  ComputeHorizonVolumeQuantities.hpp
  ComputeHorizonVolumeQuantities.tpp
  ComputeItems.hpp
  FastFlow.hpp
  Strahlkorper.hpp
  StrahlkorperGr.hpp
  StrahlkorperInDifferentFrame.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TagsTypeAliases.hpp
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
  )
