# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerSolutions")

set(LIBRARY_SOURCES
  Test_IsentropicVortex.cpp
  Test_LaneEmdenStar.cpp
  Test_RiemannProblem.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/"
  "${LIBRARY_SOURCES}"
  "NewtonianEulerSolutions;Options;Utilities"
  )
