# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizons)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ChangeCenterOfStrahlkorper.cpp
  FastFlow.cpp
  SpherepackIterator.cpp
  Strahlkorper.cpp
  StrahlkorperGr.cpp
  StrahlkorperInDifferentFrame.cpp
  Tags.cpp
  YlmSpherepack.cpp
  YlmSpherepackHelper.cpp
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
  SpherepackIterator.hpp
  Strahlkorper.hpp
  StrahlkorperGr.hpp
  StrahlkorperInDifferentFrame.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TagsTypeAliases.hpp
  YlmSpherepack.hpp
  YlmSpherepackHelper.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  DataStructures
  ErrorHandling
  GeneralRelativity
  LinearAlgebra
  Options
  SPHEREPACK
  )
