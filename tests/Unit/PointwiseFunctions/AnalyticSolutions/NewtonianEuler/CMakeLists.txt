# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerSolutions")

set(LIBRARY_SOURCES
  Test_IsentropicVortex.cpp
  Test_LaneEmdenStar.cpp
  Test_RiemannProblem.cpp
  Test_SmoothFlow.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  NewtonianEulerSolutions
  NewtonianEulerSources
  Options
  Utilities
  )
