# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarSelfForceBoundaryConditions")

set(LIBRARY_SOURCES
  Test_Angular.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainStructure
  Elliptic
  LinearOperators
  ScalarSelfForceBoundaryConditions
  Utilities
  )
