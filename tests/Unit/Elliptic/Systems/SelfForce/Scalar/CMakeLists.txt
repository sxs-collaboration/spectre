# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarSelfForce")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  ScalarSelfForce
  )

add_subdirectory(AmrCriteria)
add_subdirectory(AnalyticData)
add_subdirectory(BoundaryConditions)
add_subdirectory(Events)
