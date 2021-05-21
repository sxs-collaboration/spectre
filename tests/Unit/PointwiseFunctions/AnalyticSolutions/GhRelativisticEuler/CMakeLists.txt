# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhRelativisticEulerSolutions")

set(LIBRARY_SOURCES
  Test_InstantiateWrappedGr.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler"
  "${LIBRARY_SOURCES}"
  "GeneralRelativitySolutions;GhRelativisticEulerSolutions;RelativisticEulerSolutions;Options;Utilities"
  )
