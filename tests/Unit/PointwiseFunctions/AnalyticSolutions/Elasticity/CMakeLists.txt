# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ElasticitySolutions")

set(LIBRARY_SOURCES
  Test_BentBeam.cpp
  Test_HalfSpaceMirror.cpp
  Test_Zero.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Elasticity/"
  "${LIBRARY_SOURCES}"
  "ConstitutiveRelations;CoordinateMaps;DataStructures;Domain;Elasticity;ElasticitySolutions;Spectral;Utilities"
  )
