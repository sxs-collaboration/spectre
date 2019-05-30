# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_M1GreySolutions")

set(LIBRARY_SOURCES
  Test_ConstantM1.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey"
  "${LIBRARY_SOURCES}"
  "M1Grey;M1GreySolutions;Utilities"
  )
