# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_POINTWISE_FUNCTIONS_TESTS)

add_subdirectory(MathFunctions)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${SPECTRE_UNIT_POINTWISE_FUNCTIONS_TESTS}"
    PARENT_SCOPE)
