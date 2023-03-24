# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainAmrHelpers")

set(LIBRARY_SOURCES
  Test_NeighborFlagHelpers.cpp
  )

add_test_library(
  ${LIBRARY}
  "Helpers/Tests/Domain/Amr/"
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
