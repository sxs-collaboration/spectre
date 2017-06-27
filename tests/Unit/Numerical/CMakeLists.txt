# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_NUMERICAL_TESTS)

add_subdirectory(Spectral)
add_subdirectory(RootFinding)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${SPECTRE_UNIT_NUMERICAL_TESTS}"
    PARENT_SCOPE)
