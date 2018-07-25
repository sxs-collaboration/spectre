# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarWave")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ScalarWave/"
  "${LIBRARY_SOURCES}"
  "ScalarWave;MathFunctions;WaveEquation"
  )
