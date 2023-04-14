# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_CurvedScalarWave")

set(LIBRARY_SOURCES
  BoundaryCorrections/Test_UpwindPenalty.cpp
  BoundaryConditions/Test_ConstraintPreservingSphericalRadiation.cpp
  BoundaryConditions/Test_DemandOutgoingCharSpeeds.cpp
  BoundaryConditions/Test_Worldtube.cpp
  Test_BackgroundSpacetime.cpp
  Test_CalculateGrVars.cpp
  Test_Constraints.cpp
  Test_Characteristics.cpp
  Test_InitializeConstraintDampingGammas.cpp
  Test_InitializeEvolvedVariables.cpp
  Test_PsiSquared.cpp
  Test_Tags.cpp
  Test_TimeDerivative.cpp
  Worldtube/Test_PunctureField.cpp
  Worldtube/Test_Tags.cpp
  Worldtube/ElementActions/Test_SendToWorldtube.cpp
  Worldtube/ElementActions/Test_ReceiveWorldtubeData.cpp
  Worldtube/ElementActions/Test_InitializeConstraintGammas.cpp
  Worldtube/SingletonActions/Test_ChangeSlabSize.cpp
  Worldtube/SingletonActions/Test_InitializeElementFacesGridCoordinates.cpp
  Worldtube/SingletonActions/Test_InitializeEvolvedVariables.cpp
  Worldtube/SingletonActions/Test_InitializeSpacetimeTags.cpp
  Worldtube/SingletonActions/Test_ObserveWorldtubeSolution.cpp
  Worldtube/SingletonActions/Test_TimeDerivative.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/CurvedScalarWave/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CurvedScalarWave
  CurvedWaveEquationAnalyticData
  CurvedScalarWaveHelpers
  DataStructures
  DomainBoundaryConditions
  DomainBoundaryConditionsHelpers
  DomainCreators
  GeneralRelativityHelpers
  GeneralRelativitySolutions
  MathFunctions
  ScalarWaveWorldtube
  Utilities
  WaveEquationSolutions
)
