# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_NUMERICAL_TESTS)

add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Interpolation)
add_subdirectory(LinearOperators)
add_subdirectory(RootFinding)
add_subdirectory(Spectral)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${SPECTRE_UNIT_NUMERICAL_TESTS}"
    PARENT_SCOPE)
