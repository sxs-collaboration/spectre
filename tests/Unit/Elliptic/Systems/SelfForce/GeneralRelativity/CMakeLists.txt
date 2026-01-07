# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrSelfForce")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  GrSelfForce
  )

add_subdirectory(Actions)
add_subdirectory(AnalyticData)
add_subdirectory(BoundaryConditions)
