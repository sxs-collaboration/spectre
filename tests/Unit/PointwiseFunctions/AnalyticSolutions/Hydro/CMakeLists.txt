# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_HydroSolutions")

set(LIBRARY_SOURCES
  Test_SmoothFlow.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Hydro/"
  "${LIBRARY_SOURCES}"
  "HydroSolutions;Options;Utilities"
  )
