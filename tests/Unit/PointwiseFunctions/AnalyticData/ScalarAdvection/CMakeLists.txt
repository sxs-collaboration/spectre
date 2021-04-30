# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarAdvectionAnalyticData")

set(LIBRARY_SOURCES
  Test_Sinusoid.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/ScalarAdvection"
  "${LIBRARY_SOURCES}"
  "ScalarAdvectionAnalyticData;Utilities"
  )
