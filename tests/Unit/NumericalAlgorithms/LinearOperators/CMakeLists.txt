# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_LinearOperators")

set(LIBRARY_SOURCES
  Test_ApplyMatrices.cpp
  Test_CoefficientTransforms.cpp
  Test_DefiniteIntegral.cpp
  Test_Divergence.cpp
  Test_IndefiniteIntegral.cpp
  Test_Linearize.cpp
  Test_MeanValue.cpp
  Test_PartialDerivatives.cpp
  Test_Transpose.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/LinearOperators/"
  "${LIBRARY_SOURCES}"
  "LinearOperators;MathFunctions;Utilities"
  )
