# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_XctsPointwiseFunctions)

set(LIBRARY_SOURCES
  Test_AdmLinearMomentum.cpp
  Test_AdmMass.cpp
  Test_CenterOfMass.cpp
  Test_ExtrinsicCurvature.cpp
  Test_LongitudinalOperator.cpp
  Test_SpacetimeQuantities.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  GeneralRelativitySolutions
  LinearOperators
  ElasticityPointwiseFunctions
  GeneralRelativityHelpers
  Spectral
  Utilities
  XctsPointwiseFunctions
  XctsSolutions
  )
