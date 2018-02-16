# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_POINTWISE_FUNCTIONS)

add_subdirectory(AnalyticSolutions)
add_subdirectory(GeneralRelativity)
add_subdirectory(MathFunctions)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${SPECTRE_UNIT_POINTWISE_FUNCTIONS}"
    PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK}" PARENT_SCOPE)
