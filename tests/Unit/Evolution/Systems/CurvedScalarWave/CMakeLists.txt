# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Worldtube)

set(LIBRARY "Test_CurvedScalarWave")

set(LIBRARY_SOURCES
  BoundaryCorrections/Test_UpwindPenalty.cpp
  BoundaryConditions/Test_AnalyticConstant.cpp
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
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CurvedScalarWave
  CurvedWaveEquationAnalyticData
  CurvedScalarWaveHelpers
  DataStructures
  DomainBoundaryConditions
  DomainBoundaryConditionsHelpers
  GeneralRelativityHelpers
  MathFunctions
  Time
  Utilities
  WaveEquationSolutions
)
