# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Poisson")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Poisson/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  EllipticTestHelpers
  Poisson
  Utilities
  )

add_subdirectory(BoundaryConditions)
