# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_XctsAnalyticData")

set(LIBRARY_SOURCES
  Test_Binary.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/Xcts/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Utilities
  Xcts
  XctsAnalyticData
  XctsSolutions
  )
