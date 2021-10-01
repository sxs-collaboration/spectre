# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainStructure")

set(LIBRARY_SOURCES
  Test_BlockId.cpp
  Test_BlockNeighbor.cpp
  Test_ChildSize.cpp
  Test_CreateInitialMesh.cpp
  Test_Direction.cpp
  Test_Element.cpp
  Test_ElementId.cpp
  Test_ExcisionSphere.cpp
  Test_Hypercube.cpp
  Test_IndexToSliceAt.cpp
  Test_InitialElementIds.cpp
  Test_Neighbors.cpp
  Test_OrientationMap.cpp
  Test_OrientationMapHelpers.cpp
  Test_SegmentId.cpp
  Test_Side.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Structure"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainHelpers
  DomainStructure
  Utilities
  )
