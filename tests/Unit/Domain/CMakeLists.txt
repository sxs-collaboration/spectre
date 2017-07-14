# Distributed under the MIT License.
# See LICENSE.txt for details.

set(DOMAIN_TESTS
    Domain/Test_Direction.cpp
    Domain/Test_EmbeddingMaps.cpp
    Domain/Test_ElementId.cpp
    Domain/Test_Orientations.cpp
    Domain/Test_SegmentId.cpp
    Domain/Test_Side.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${DOMAIN_TESTS}" PARENT_SCOPE)
