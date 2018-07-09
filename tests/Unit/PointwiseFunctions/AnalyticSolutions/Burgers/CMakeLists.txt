# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_BurgersSolutions")

set(LIBRARY_SOURCES
  Test_Bump.cpp
  Test_Linear.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Burgers/"
  "${LIBRARY_SOURCES}"
  "BurgersSolutions;Burgers;Utilities"
  )
