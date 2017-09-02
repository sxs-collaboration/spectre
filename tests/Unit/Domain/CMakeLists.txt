# Distributed under the MIT License.
# See LICENSE.txt for details.

set(DOMAIN_TESTS
    Domain/CoordinateMaps/Test_AffineMap.cpp
    Domain/CoordinateMaps/Test_CoordinateMap.cpp
    Domain/CoordinateMaps/Test_Identity.cpp
    Domain/CoordinateMaps/Test_ProductMaps.cpp
    Domain/CoordinateMaps/Test_Rotation.cpp
    Domain/Test_Block.cpp
    Domain/Test_BlockNeighbor.cpp
    Domain/Test_Direction.cpp
    Domain/Test_ElementId.cpp
    Domain/Test_GridCoordinates.cpp
    Domain/Test_Neighbors.cpp
    Domain/Test_Orientations.cpp
    Domain/Test_SegmentId.cpp
    Domain/Test_Side.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${DOMAIN_TESTS}" PARENT_SCOPE)
