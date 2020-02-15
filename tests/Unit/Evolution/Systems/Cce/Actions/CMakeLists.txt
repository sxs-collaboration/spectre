# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce_Actions")

set(LIBRARY_SOURCES
  Test_BoundaryCommunication.cpp
  Test_CharacteristicEvolutionBondiCalculations.cpp
  Test_FilterSwshVolumeQuantity.cpp
  Test_InitializeCharacteristicEvolution.cpp
  Test_InitializeWorldtubeBoundary.cpp
  Test_RequestBoundaryData.cpp
  Test_TimeManagement.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/Actions/"
  "${LIBRARY_SOURCES}"
  "Cce"
  )
