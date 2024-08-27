# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(EquationsOfState)
add_subdirectory(InitialData)
add_subdirectory(Python)

set(LIBRARY "Test_Hydro")

set(LIBRARY_SOURCES
  Test_ComovingMagneticField.cpp
  Test_InversePlasmaBeta.cpp
  Test_LorentzFactor.cpp
  Test_MagneticFieldTreatment.cpp
  Test_MassFlux.cpp
  Test_MassWeightedFluidItems.cpp
  Test_QuadrupoleFormula.cpp
  Test_SoundSpeedSquared.cpp
  Test_SpecificEnthalpy.cpp
  Test_StressEnergy.cpp
  Test_Tags.cpp
  Test_Temperature.cpp
  Test_TestHelpers.cpp
  Test_TransportVelocity.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DataStructuresHelpers
  Domain
  GeneralRelativity
  GeneralRelativityHelpers
  Hydro
  HydroHelpers
  LinearOperators
  Spectral
  IO
  Utilities
)
