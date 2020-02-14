# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(EquationsOfState)

set(LIBRARY "Test_Hydro")

set(LIBRARY_SOURCES
  TestHelpers.cpp
  Test_LorentzFactor.cpp
  Test_MassFlux.cpp
  Test_SoundSpeedSquared.cpp
  Test_SpecificEnthalpy.cpp
  Test_Tags.cpp
  Test_TestHelpers.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/Hydro/"
  "${LIBRARY_SOURCES}"
  "Test_Pypp;Hydro"
  )
