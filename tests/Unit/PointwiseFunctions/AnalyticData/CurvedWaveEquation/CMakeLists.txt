# Distributed under the MIT License.
# See LICENSE.txt for details.


set(LIBRARY "Test_CurvedWaveEquationAnalyticData")

set(LIBRARY_SOURCES
  Test_ScalarWaveKerrSchild.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/CurvedWaveEquation/"
  "${LIBRARY_SOURCES}"
  "CurvedWaveEquationAnalyticData;GeneralRelativitySolutions;GeneralRelativity;MathFunctions;Utilities;WaveEquation"
  )
