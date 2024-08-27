# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarGaussBonnetBoundaryConditions")

set(LIBRARY_SOURCES
  Test_DoNothing.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Elliptic
  ScalarGaussBonnet
  Utilities
  )
