# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerSolutions")

set(LIBRARY_SOURCES
  Test_IsentropicVortex.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/"
  "${LIBRARY_SOURCES}"
  "NewtonianEulerSolutions;Utilities"
  )
