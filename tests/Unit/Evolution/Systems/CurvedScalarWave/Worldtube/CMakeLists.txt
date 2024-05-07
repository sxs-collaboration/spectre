# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarWaveWorldtube")

set(LIBRARY_SOURCES
  Test_AccelerationTerms.cpp
  Test_KerrSchildDerivatives.cpp
  Test_PunctureField.cpp
  Test_SelfForce.cpp
  Test_Tags.cpp
  ElementActions/Test_SendToWorldtube.cpp
  ElementActions/Test_ReceiveWorldtubeData.cpp
  ElementActions/Test_InitializeConstraintGammas.cpp
  ElementActions/Test_InitializeCurrentIteration.cpp
  ElementActions/Test_Iterations.cpp
  SingletonActions/Test_ChangeSlabSize.cpp
  SingletonActions/Test_InitializeElementFacesGridCoordinates.cpp
  SingletonActions/Test_InitializeEvolvedVariables.cpp
  SingletonActions/Test_ObserveWorldtubeSolution.cpp
  SingletonActions/Test_UpdateAcceleration.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CurvedScalarWave
  CurvedScalarWaveHelpers
  DataStructures
  DomainBoundaryConditions
  DomainBoundaryConditionsHelpers
  DomainCreators
  GeneralRelativity
  GeneralRelativityHelpers
  GeneralRelativitySolutions
  ObserverHelpers
  ScalarWaveWorldtube
  Time
  Utilities
)
