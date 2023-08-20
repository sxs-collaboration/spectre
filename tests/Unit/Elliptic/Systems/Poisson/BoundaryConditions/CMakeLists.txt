# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PoissonBoundaryConditions")

set(LIBRARY_SOURCES
  Test_Robin.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainStructure
  Elliptic
  PoissonBoundaryConditions
  Utilities
  )
