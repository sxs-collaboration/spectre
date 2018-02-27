# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_DOMAIN
  Domain/DomainTestHelpers.cpp
  Domain/Test_Block.cpp
  Domain/Test_BlockNeighbor.cpp
  Domain/Test_CoordinatesTag.cpp
  Domain/Test_CreateInitialElement.cpp
  Domain/Test_Direction.cpp
  Domain/Test_Domain.cpp
  Domain/Test_DomainHelpers.cpp
  Domain/Test_Element.cpp
  Domain/Test_ElementId.cpp
  Domain/Test_ElementIndex.cpp
  Domain/Test_ElementMap.cpp
  Domain/Test_FaceNormal.cpp
  Domain/Test_InitialElementIds.cpp
  Domain/Test_InterfaceItems.cpp
  Domain/Test_LogicalCoordinates.cpp
  Domain/Test_Neighbors.cpp
  Domain/Test_OrientationMap.cpp
  Domain/Test_SegmentId.cpp
  Domain/Test_Side.cpp
  )

add_subdirectory(CoordinateMaps)
add_subdirectory(DomainCreators)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${SPECTRE_UNIT_DOMAIN}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};Domain" PARENT_SCOPE)
