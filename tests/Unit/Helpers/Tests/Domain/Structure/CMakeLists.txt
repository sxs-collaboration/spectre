# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainStructureHelpers")

set(LIBRARY_SOURCES
  Test_NeighborHelpers.cpp
  Test_OrientationMapHelpers.cpp
  )

add_test_library(
  ${LIBRARY}
  "Helpers/Tests/Domain/Structure/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainStructure
  DomainStructureHelpers
  )
