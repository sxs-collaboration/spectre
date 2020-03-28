# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Amr)
add_subdirectory(CoordinateMaps)
add_subdirectory(Creators)
add_subdirectory(FunctionsOfTime)
add_subdirectory(Python)

set(LIBRARY Domain)

set(LIBRARY_SOURCES
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
  ElementIndex.cpp
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

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  FunctionsOfTime
  Time

  # PUBLIC
  Boost::boost
  CoordinateMaps
  DataStructures
  ErrorHandling
  Options
  Utilities

  # PRIVATE
  DomainCreators
  Spectral
  )
