# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_BurgersAnalyticData")

set(LIBRARY_SOURCES
  Test_Sinusoid.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/Burgers"
  "${LIBRARY_SOURCES}"
  "BurgersAnalyticData;Utilities"
  )
