# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ConstitutiveRelations")

set(LIBRARY_SOURCES
  Test_CubicCrystal.cpp
  Test_IsotropicHomogeneous.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstitutiveRelations
  DataStructures
  Utilities
  )
