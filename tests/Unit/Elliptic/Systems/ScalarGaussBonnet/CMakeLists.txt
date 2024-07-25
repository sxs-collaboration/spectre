# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarGaussBonnet")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  Test_Equations.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  EllipticTestHelpers
  ScalarGaussBonnet
  Utilities
  )

add_subdirectory(BoundaryConditions)
