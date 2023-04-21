# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Amr")

set(LIBRARY_SOURCES
  Test_Flag.cpp
  Test_Helpers.cpp
  Test_NeighborsOfChild.cpp
  Test_NeighborsOfParent.cpp
  Test_NewNeighborIds.cpp
  Test_Tags.cpp
  Test_UpdateAmrDecision.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Amr"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  DomainAmrHelpers
  DomainStructure
  DomainStructureHelpers
  Utilities
  )
