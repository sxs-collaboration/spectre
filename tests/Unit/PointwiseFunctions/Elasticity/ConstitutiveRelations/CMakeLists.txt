# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ConstitutiveRelations")

set(LIBRARY_SOURCES
  Test_IsotropicHomogeneous.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/Elasticity/ConstitutiveRelations"
  "${LIBRARY_SOURCES}"
  "ConstitutiveRelations;DataStructures;Utilities"
  )
