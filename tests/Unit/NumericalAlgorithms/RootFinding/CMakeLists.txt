# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_RootFinding")

set(LIBRARY_SOURCES
  Test_OneDRootFinder.cpp
  Test_QuadraticEquation.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/RootFinding/"
  "${LIBRARY_SOURCES}"
  "RootFinding"
  )
