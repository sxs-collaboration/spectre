# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_XctsSolutions")

set(LIBRARY_SOURCES
  Test_ConstantDensityStar.cpp
  Test_Schwarzschild.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Xcts/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities;Xcts;XctsSolutions"
  )
