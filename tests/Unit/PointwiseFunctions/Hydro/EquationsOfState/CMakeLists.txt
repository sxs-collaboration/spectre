# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EquationsOfState")

set(LIBRARY_SOURCES
  Test_DarkEnergyFluid.cpp
  Test_IdealFluid.cpp
  Test_PolytropicFluid.cpp
  Test_SpecificEnthalpy.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/Hydro/EquationsOfState/"
  "${LIBRARY_SOURCES}"
  "EquationsOfState;DataStructures;Utilities"
  )
