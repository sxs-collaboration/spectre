# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_ACTIONS_TESTS
    Numerical/DiscontinuousGalerkin/Actions/Test_FluxCommunication.cpp
   )

set(SPECTRE_UNIT_DISCONTINUOUS_GALERKIN_TESTS
    "${SPECTRE_UNIT_DISCONTINUOUS_GALERKIN_TESTS};${SPECTRE_UNIT_ACTIONS_TESTS}"
    PARENT_SCOPE)
