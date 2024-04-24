# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce_Actions")

set(LIBRARY_SOURCES
  Test_AnalyticBoundaryCommunication.cpp
  Test_CalculateScriInputs.cpp
  Test_CharacteristicEvolutionBondiCalculations.cpp
  Test_InitializeFirstHypersurface.cpp
  Test_InsertInterpolationScriData.cpp
  Test_FilterSwshVolumeQuantity.cpp
  Test_GhBoundaryCommunication.cpp
  Test_H5BoundaryCommunication.cpp
  Test_InitializeCharacteristicEvolution.cpp
  Test_InitializeKleinGordonCharacteristicEvolution.cpp
  Test_InitializeKleinGordonFirstHypersurface.cpp
  Test_InitializeKleinGordonWorldtubeBoundary.cpp
  Test_InitializeWorldtubeBoundary.cpp
  Test_KleinGordonCceCalculations.cpp
  Test_KleinGordonH5BoundaryCommunication.cpp
  Test_Psi0Matching.cpp
  Test_RequestBoundaryData.cpp
  Test_RequestKleinGordonBoundaryData.cpp
  Test_SendGhVarsToCce.cpp
  Test_ScriObserveInterpolated.cpp
  Test_TimeManagement.cpp
  Test_UpdateGauge.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Cce
  CceHelpers
  GeneralRelativitySolutions
  Interpolation
  InterpolationHelpers
  SpinWeightedSphericalHarmonics
  )
