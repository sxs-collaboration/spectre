# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_RelativisticEulerSolutions")

set(LIBRARY_SOURCES
  Test_FishboneMoncriefDisk.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
  "${LIBRARY_SOURCES}"
  "RelativisticEulerSolutions;Utilities"
  )
