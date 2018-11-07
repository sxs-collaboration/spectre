# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralRelativitySolutions")

set(LIBRARY_SOURCES
  Test_KerrHorizon.cpp
  Test_KerrSchild.cpp
  Test_Minkowski.cpp
  Test_Tov.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/"
  "${LIBRARY_SOURCES}"
  "Test_Pypp;GeneralRelativitySolutions;GeneralizedHarmonic;GeneralRelativity;Utilities"
  )
