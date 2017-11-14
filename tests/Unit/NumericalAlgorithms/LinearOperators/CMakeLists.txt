# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_LINEAR_OPERATORS
  NumericalAlgorithms/LinearOperators/Test_DefiniteIntegral.cpp
  NumericalAlgorithms/LinearOperators/Test_Linearize.cpp
  NumericalAlgorithms/LinearOperators/Test_MeanValue.cpp
  NumericalAlgorithms/LinearOperators/Test_PartialDerivatives.cpp
  NumericalAlgorithms/LinearOperators/Test_Transpose.cpp
  )

set(SPECTRE_UNIT_NUMERICAL_ALGORITHMS
  "${SPECTRE_UNIT_NUMERICAL_ALGORITHMS};${SPECTRE_UNIT_LINEAR_OPERATORS}"
  PARENT_SCOPE
  )
