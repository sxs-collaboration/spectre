# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DgMessages")

set(LIBRARY_SOURCES
  Test_BoundaryMessage.cpp
  Test_InboxTags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  ""
  WITH_CHARM
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructuresHelpers
  DiscontinuousGalerkin
  DomainStructure
  Evolution
  Spectral
  Time
  Utilities
  )
