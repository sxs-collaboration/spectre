# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(FunctionsOfTime)
add_subdirectory(Python)

set(LIBRARY "Test_Domain")

set(LIBRARY_SOURCES
  Structure/Test_Hypercube.cpp
  Test_Block.cpp
  Test_BlockAndElementLogicalCoordinates.cpp
  Test_BlockId.cpp
  Test_BlockNeighbor.cpp
  Test_CoordinatesTag.cpp
  Test_CreateInitialElement.cpp
  Test_CreateInitialMesh.cpp
  Test_Direction.cpp
  Test_Domain.cpp
  Test_DomainHelpers.cpp
  Test_DomainTestHelpers.cpp
  Test_Element.cpp
  Test_ElementId.cpp
  Test_ElementMap.cpp
  Test_FaceNormal.cpp
  Test_IndexToSliceAt.cpp
  Test_InitialElementIds.cpp
  Test_InterfaceHelpers.cpp
  Test_InterfaceItems.cpp
  Test_LogicalCoordinates.cpp
  Test_Mesh.cpp
  Test_MinimumGridSpacing.cpp
  Test_Neighbors.cpp
  Test_OrientationMap.cpp
  Test_OrientationMapHelpers.cpp
  Test_SegmentId.cpp
  Test_Side.cpp
  Test_SizeOfElement.cpp
  Test_Tags.cpp
  Test_TagsCharacteristicSpeeds.cpp
  Test_TagsTimeDependent.cpp
  )

add_subdirectory(Amr)
add_subdirectory(CoordinateMaps)
add_subdirectory(Creators)

add_test_library(
  ${LIBRARY}
  "Domain"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Domain
  DomainCreators
  DomainHelpers
  Utilities
  )
