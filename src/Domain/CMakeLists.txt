# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Domain)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Block.cpp
  BlockLogicalCoordinates.cpp
  BlockNeighbor.cpp
  CreateInitialElement.cpp
  CreateInitialMesh.cpp
  Direction.cpp
  Domain.cpp
  DomainHelpers.cpp
  Element.cpp
  ElementId.cpp
  ElementLogicalCoordinates.cpp
  ElementMap.cpp
  FaceNormal.cpp
  InitialElementIds.cpp
  LogicalCoordinates.cpp
  Mesh.cpp
  MinimumGridSpacing.cpp
  Neighbors.cpp
  OrientationMap.cpp
  OrientationMapHelpers.cpp
  SegmentId.cpp
  Side.cpp
  SizeOfElement.cpp
  Tags.cpp
  TagsTimeDependent.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Block.hpp
  BlockId.hpp
  BlockLogicalCoordinates.hpp
  BlockNeighbor.hpp
  CreateInitialElement.hpp
  CreateInitialMesh.hpp
  Direction.hpp
  DirectionMap.hpp
  Domain.hpp
  DomainHelpers.hpp
  Element.hpp
  ElementId.hpp
  ElementLogicalCoordinates.hpp
  ElementMap.hpp
  FaceNormal.hpp
  IndexToSliceAt.hpp
  InitialElementIds.hpp
  InterfaceComputeTags.hpp
  InterfaceHelpers.hpp
  LogicalCoordinates.hpp
  MaxNumberOfNeighbors.hpp
  Mesh.hpp
  MinimumGridSpacing.hpp
  Neighbors.hpp
  OptionTags.hpp
  OrientationMap.hpp
  OrientationMapHelpers.hpp
  SegmentId.hpp
  Side.hpp
  SizeOfElement.hpp
  Tags.hpp
  TagsCharacteresticSpeeds.hpp
  TagsTimeDependent.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  Options
  Utilities
  INTERFACE
  FunctionsOfTime
  Time
  CoordinateMaps
  DomainCreators
  Spectral
  )

add_subdirectory(Amr)
add_subdirectory(CoordinateMaps)
add_subdirectory(Creators)
add_subdirectory(FunctionsOfTime)
add_subdirectory(Python)
