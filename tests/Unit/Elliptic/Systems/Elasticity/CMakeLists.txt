# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Elasticity")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstitutiveRelations
  DataStructures
  Domain
  Elliptic
  Elasticity
  LinearOperators
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
