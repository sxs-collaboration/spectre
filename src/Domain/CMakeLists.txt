# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Amr)
add_subdirectory(CoordinateMaps)
add_subdirectory(DomainCreators)

set(LIBRARY Domain)

set(LIBRARY_SOURCES
    Block.cpp
    BlockNeighbor.cpp
    CreateInitialElement.cpp
    Domain.cpp
    DomainHelpers.cpp
    Direction.cpp
    Element.cpp
    ElementId.cpp
    ElementIndex.cpp
    ElementMap.cpp
    FaceNormal.cpp
    InitialElementIds.cpp
    LogicalCoordinates.cpp
    Mesh.cpp
    Neighbors.cpp
    OrientationMap.cpp
    SegmentId.cpp
    Side.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE CoordinateMaps
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Spectral
  )
