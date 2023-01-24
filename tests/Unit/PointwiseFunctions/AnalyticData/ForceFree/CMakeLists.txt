# Distributed under the MIT License.
# See LICENSE.txt for details.


set(LIBRARY "Test_ForceFreeAnalyticData")

set(LIBRARY_SOURCES
  Test_FfeBreakdown.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/ForceFree"
  "${LIBRARY_SOURCES}"
  "ForceFreeAnalyticData;Options;Utilities"
  )
