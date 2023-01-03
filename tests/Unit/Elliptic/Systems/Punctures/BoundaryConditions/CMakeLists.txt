# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PuncturesBoundaryConditions")

set(LIBRARY_SOURCES
  Test_Flatness.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Punctures/BoundaryConditions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainStructure
  Elliptic
  PuncturesBoundaryConditions
  Utilities
  )
