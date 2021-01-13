# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce_Actions")

set(LIBRARY_SOURCES
  Test_CalculateScriInputs.cpp
  Test_CharacteristicEvolutionBondiCalculations.cpp
  Test_InitializeFirstHypersurface.cpp
  Test_InsertInterpolationScriData.cpp
  Test_FilterSwshVolumeQuantity.cpp
  Test_GhBoundaryCommunication.cpp
  Test_H5BoundaryCommunication.cpp
  Test_InitializeCharacteristicEvolution.cpp
  Test_InitializeWorldtubeBoundary.cpp
  Test_RequestBoundaryData.cpp
  Test_ScriObserveInterpolated.cpp
  Test_SendNextTimeToCce.cpp
  Test_TimeManagement.cpp
  Test_UpdateGauge.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/Actions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Cce
  GeneralRelativitySolutions
  Interpolation
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
