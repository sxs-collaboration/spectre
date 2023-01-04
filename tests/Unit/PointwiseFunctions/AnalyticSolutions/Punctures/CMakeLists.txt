# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PuncturesSolutions")

set(LIBRARY_SOURCES
  Test_Flatness.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Punctures/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  PuncturesSolutions
  )
