# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarAdvectionSolutions")

set(LIBRARY_SOURCES
  Test_Krivodonova.cpp
  Test_Sinusoid.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/"
  "${LIBRARY_SOURCES}"
  "ScalarAdvectionSolutions;Utilities"
  )
