# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_INTERPOLATION
    NumericalAlgorithms/Interpolation/Test_LagrangePolynomial.cpp
    )

set(SPECTRE_UNIT_NUMERICAL_ALGORITHMS
    "${SPECTRE_UNIT_NUMERICAL_ALGORITHMS};${SPECTRE_UNIT_INTERPOLATION}"
    PARENT_SCOPE)
