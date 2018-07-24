# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EinsteinSolutions")

set(LIBRARY_SOURCES
  Test_KerrSchild.cpp
  Test_Minkowski.cpp
  Test_Tov.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/"
  "${LIBRARY_SOURCES}"
  "EinsteinSolutions;GeneralizedHarmonic;GeneralRelativity;Utilities"
  )
