# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_SPECTRAL_TESTS
    Numerical/Spectral/Test_DefiniteIntegral.cpp
    Numerical/Spectral/Test_LegendreGaussLobatto.cpp
    Numerical/Spectral/Test_Linearize.cpp
    )

set(SPECTRE_UNIT_NUMERICAL_TESTS
    "${SPECTRE_UNIT_NUMERICAL_TESTS};${SPECTRE_UNIT_SPECTRAL_TESTS}"
    PARENT_SCOPE)
