# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralRelativityAnalyticData")

set(LIBRARY_SOURCES
  Test_BrillLindquist.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/GeneralRelativity"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GeneralRelativityAnalyticData
  DataStructures
  DataStructuresHelpers
  )
