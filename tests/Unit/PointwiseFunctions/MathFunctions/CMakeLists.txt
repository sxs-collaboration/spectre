# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_FUNCTION_TESTS
    PointwiseFunctions/MathFunctions/Test_Gaussian.cpp
    PointwiseFunctions/MathFunctions/Test_PowX.cpp
   )

set(SPECTRE_UNIT_POINTWISE_FUNCTIONS_TESTS
    "${SPECTRE_UNIT_POINTWISE_FUNCTIONS_TESTS};${SPECTRE_UNIT_FUNCTION_TESTS}"
    PARENT_SCOPE)
