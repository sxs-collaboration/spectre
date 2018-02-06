# Distributed under the MIT License.
# See LICENSE.txt for details.

set(DOMAIN_TESTS
    Domain/CoordinateMaps/Test_Affine.cpp
    Domain/CoordinateMaps/Test_BulgedCube.cpp
    Domain/CoordinateMaps/Test_CoordinateMap.cpp
    Domain/CoordinateMaps/Test_Equiangular.cpp
    Domain/CoordinateMaps/Test_Identity.cpp
    Domain/CoordinateMaps/Test_ProductMaps.cpp
    Domain/CoordinateMaps/Test_Rotation.cpp
    Domain/CoordinateMaps/Test_Wedge2D.cpp
    Domain/CoordinateMaps/Test_Wedge3D.cpp
    Domain/DomainCreators/Test_Brick.cpp
    Domain/DomainCreators/Test_Disk.cpp
    Domain/DomainCreators/Test_Interval.cpp
    Domain/DomainCreators/Test_Rectangle.cpp
    Domain/DomainCreators/Test_Shell.cpp
    Domain/DomainCreators/Test_Sphere.cpp
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
    Domain/Test_LogicalCoordinates.cpp
    Domain/Test_Neighbors.cpp
    Domain/Test_OrientationMap.cpp
    Domain/Test_SegmentId.cpp
    Domain/Test_Side.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${DOMAIN_TESTS}" PARENT_SCOPE)
