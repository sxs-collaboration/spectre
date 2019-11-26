# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce")

set(LIBRARY_SOURCES
  CceComputationTestHelpers.cpp
  Test_BoundaryData.cpp
  Test_Equations.cpp
  Test_InitializeCce.cpp
  Test_LinearOperators.cpp
  Test_LinearSolve.cpp
  Test_PreSwshDerivatives.cpp
  Test_PrecomputeCceDependencies.cpp
  Test_SwshDerivatives.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/"
  "${LIBRARY_SOURCES}"
  "Cce"
  )
