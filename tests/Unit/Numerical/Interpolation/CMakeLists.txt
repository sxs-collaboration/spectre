# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_INTERPOLATION_TESTS
    Numerical/Interpolation/Test_LagrangePolynomial.cpp
    )

set(SPECTRE_UNIT_NUMERICAL_TESTS
    "${SPECTRE_UNIT_NUMERICAL_TESTS};${SPECTRE_UNIT_INTERPOLATION_TESTS}"
    PARENT_SCOPE)
