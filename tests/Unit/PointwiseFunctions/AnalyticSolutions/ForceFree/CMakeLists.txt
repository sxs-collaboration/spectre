# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ForceFreeSolutions")

set(LIBRARY_SOURCES
  Test_AlfvenWave.cpp
  Test_FastWave.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/ForceFree"
  "${LIBRARY_SOURCES}"
  "ForceFreeSolutions;Options;Utilities"
  )
