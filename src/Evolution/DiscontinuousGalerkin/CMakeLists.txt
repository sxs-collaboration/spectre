# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DgElementArray.hpp
  InboxTags.hpp
  InterpolateFromBoundary.hpp
  LiftFromBoundary.hpp
  MortarData.hpp
  MortarTags.hpp
  NormalVectorTags.hpp
  ProjectToBoundary.hpp
  UsingSubcell.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InterpolateFromBoundary.cpp
  LiftFromBoundary.cpp
  MortarData.cpp
  )

add_subdirectory(Initialization)
add_subdirectory(Limiters)
