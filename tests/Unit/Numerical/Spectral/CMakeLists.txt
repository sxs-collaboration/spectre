# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_SPECTRAL_TESTS
    Numerical/Spectral/LegendreGaussLobatto.cpp)

set(SPECTRE_UNIT_NUMERICAL_TESTS
    "${SPECTRE_UNIT_NUMERICAL_TESTS};${SPECTRE_UNIT_SPECTRAL_TESTS}"
    PARENT_SCOPE)
