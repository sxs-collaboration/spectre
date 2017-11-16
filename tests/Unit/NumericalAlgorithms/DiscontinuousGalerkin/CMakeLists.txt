# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_DISCONTINUOUS_GALERKIN
    NumericalAlgorithms/DiscontinuousGalerkin/Test_LiftFlux.cpp
   )

add_subdirectory(Actions)

set(SPECTRE_UNIT_NUMERICAL_ALGORITHMS
  "${SPECTRE_UNIT_NUMERICAL_ALGORITHMS};${SPECTRE_UNIT_DISCONTINUOUS_GALERKIN}"
  PARENT_SCOPE)
