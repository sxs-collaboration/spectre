# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Hydro")

set(LIBRARY_SOURCES
  EquationsOfState/Test_DarkEnergyFluid.cpp
  EquationsOfState/Test_IdealFluid.cpp
  EquationsOfState/Test_PolytropicFluid.cpp
  Test_LorentzFactor.cpp
  Test_SpecificEnthalpy.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/Hydro/"
  "${LIBRARY_SOURCES}"
  "Test_Pypp;Hydro"
  )
