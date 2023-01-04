# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PuncturesAnalyticData")

set(LIBRARY_SOURCES
  Test_MultiplePunctures.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/Punctures/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Utilities
  Punctures
  PuncturesAnalyticData
  )
