# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_LINEAR_OPERATORS_TESTS
  Numerical/LinearOperators/Test_Transpose.cpp
  )

set(SPECTRE_UNIT_NUMERICAL_TESTS
  "${SPECTRE_UNIT_NUMERICAL_TESTS};${SPECTRE_UNIT_LINEAR_OPERATORS_TESTS}"
  PARENT_SCOPE
  )
