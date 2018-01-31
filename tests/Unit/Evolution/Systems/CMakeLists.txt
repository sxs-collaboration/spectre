# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(CurvedScalarWave)
add_subdirectory(GeneralizedHarmonic)
add_subdirectory(ScalarWave)

set(EVOLUTION_TESTS "${EVOLUTION_TESTS};${SYSTEM_TESTS}" PARENT_SCOPE)
