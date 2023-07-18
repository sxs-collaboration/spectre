# Distributed under the MIT License.
# See LICENSE.txt for details.


set(LIBRARY "Test_CurvedWaveEquationAnalyticData")

set(LIBRARY_SOURCES
  Test_PureSphericalHarmonic.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/CurvedWaveEquation/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CurvedWaveEquationAnalyticData
  GeneralRelativitySolutions
  GeneralRelativity
  MathFunctions
  Utilities
  WaveEquationSolutions
  )
