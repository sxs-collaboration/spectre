# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_DISCONTINUOUS_GALERKIN_TESTS
    Numerical/DiscontinuousGalerkin/Test_LiftFlux.cpp
   )

add_subdirectory(Actions)

set(SPECTRE_UNIT_NUMERICAL_TESTS "${SPECTRE_UNIT_NUMERICAL_TESTS};${SPECTRE_UNIT_DISCONTINUOUS_GALERKIN_TESTS}"
    PARENT_SCOPE)
