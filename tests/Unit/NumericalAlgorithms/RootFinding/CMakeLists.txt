# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_RootFinding")

set(LIBRARY_SOURCES
  Test_GslMultiRoot.cpp
  Test_NewtonRaphson.cpp
  Test_QuadraticEquation.cpp
  Test_TOMS748.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/RootFinding/"
  "${LIBRARY_SOURCES}"
  "RootFinding;DataStructures"
  )
