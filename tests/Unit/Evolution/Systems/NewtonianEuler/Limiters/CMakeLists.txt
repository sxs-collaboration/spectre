# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerLimiters")

set(LIBRARY_SOURCES
  Test_CharacteristicHelpers.cpp
  Test_Flattener.cpp
  Test_KxrcfTci.cpp
  Test_Minmod.cpp
  Test_VariablesToLimit.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/NewtonianEuler/Limiters"
  "${LIBRARY_SOURCES}"
  "NewtonianEulerLimiters"
  )
