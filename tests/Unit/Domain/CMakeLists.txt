# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Domain")

set(LIBRARY_SOURCES
  DomainTestHelpers.cpp
  Test_Block.cpp
  Test_BlockAndElementLogicalCoordinates.cpp
  Test_BlockId.cpp
  Test_BlockNeighbor.cpp
  Test_CoordinatesTag.cpp
  Test_CreateInitialElement.cpp
  Test_Direction.cpp
  Test_Domain.cpp
  Test_DomainHelpers.cpp
  Test_DomainTestHelpers.cpp
  Test_Element.cpp
  Test_ElementId.cpp
  Test_ElementIndex.cpp
  Test_ElementMap.cpp
  Test_FaceNormal.cpp
  Test_IndexToSliceAt.cpp
  Test_InitialElementIds.cpp
  Test_InterfaceItems.cpp
  Test_LogicalCoordinates.cpp
  Test_Mesh.cpp
  Test_MinimumGridSpacing.cpp
  Test_Neighbors.cpp
  Test_OrientationMap.cpp
  Test_SegmentId.cpp
  Test_Side.cpp
  Test_SizeOfElement.cpp
  )

add_subdirectory(Amr)
add_subdirectory(CoordinateMaps)
add_subdirectory(Creators)

add_test_library(
  ${LIBRARY}
  "Domain"
  "${LIBRARY_SOURCES}"
  "Domain;DomainCreators;Utilities"
  )
