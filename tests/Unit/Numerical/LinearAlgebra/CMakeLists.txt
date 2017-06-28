# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_LINEAR_ALGEBRA_TESTS
    Numerical/LinearAlgebra/Test_Determinant.cpp
   )

set(SPECTRE_UNIT_NUMERICAL_TESTS
   "${SPECTRE_UNIT_NUMERICAL_TESTS};${SPECTRE_UNIT_LINEAR_ALGEBRA_TESTS}"
   PARENT_SCOPE)
