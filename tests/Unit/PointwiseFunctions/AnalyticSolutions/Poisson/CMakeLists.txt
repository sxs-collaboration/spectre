# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PoissonSolutions")

set(LIBRARY_SOURCES
  Test_Moustache.cpp
  Test_ProductOfSinusoids.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Poisson/"
  "${LIBRARY_SOURCES}"
  "PoissonSolutions;Utilities"
  )
