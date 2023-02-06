# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Punctures")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Punctures/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructuresHelpers
  Punctures
  )

add_subdirectory(AmrCriteria)
add_subdirectory(BoundaryConditions)
