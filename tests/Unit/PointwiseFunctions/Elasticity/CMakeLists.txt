# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(ConstitutiveRelations)

set(LIBRARY Test_ElasticityPointwiseFunctions)

set(LIBRARY_SOURCES
  Test_PotentialEnergy.cpp
  Test_Strain.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/Elasticity/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstitutiveRelations
  CoordinateMaps
  DataStructures
  Elasticity
  ElasticityPointwiseFunctions
  Utilities
  )
