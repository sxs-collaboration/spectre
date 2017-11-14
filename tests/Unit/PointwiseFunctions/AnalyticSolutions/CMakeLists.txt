# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(EinsteinSolutions)
add_subdirectory(WaveEquation)

set(SPECTRE_UNIT_POINTWISE_FUNCTIONS
  "${SPECTRE_UNIT_POINTWISE_FUNCTIONS};${SPECTRE_UNIT_ANALYTIC_SOLUTIONS}"
  PARENT_SCOPE)
