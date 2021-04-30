# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ElasticityActions")

set(LIBRARY_SOURCES
  Test_InitializeConstitutiveRelation.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstitutiveRelations
  DataStructures
  Domain
  ElasticityActions
  Elliptic
  Elasticity
  Utilities
  )
