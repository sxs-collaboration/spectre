# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_XctsBoundaryConditions")

set(LIBRARY_SOURCES
  Test_ApparentHorizon.cpp
  Test_Flatness.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Xcts/BoundaryConditions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainStructure
  Elliptic
  Utilities
  XctsBoundaryConditions
  XctsSolutions
  )
