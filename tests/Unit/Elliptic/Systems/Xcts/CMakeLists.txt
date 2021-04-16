# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Xcts")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Xcts/"
  "${LIBRARY_SOURCES}"
  "DataStructuresHelpers;Xcts"
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Utilities
  Xcts
  )

add_subdirectory(BoundaryConditions)
