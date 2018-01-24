# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_ACTIONS
  NumericalAlgorithms/DiscontinuousGalerkin/Actions/Test_FluxCommunication.cpp
  NumericalAlgorithms/DiscontinuousGalerkin/Actions/Test_FluxCommunication2.cpp
   )

set(SPECTRE_UNIT_DISCONTINUOUS_GALERKIN
    "${SPECTRE_UNIT_DISCONTINUOUS_GALERKIN};${SPECTRE_UNIT_ACTIONS}"
    PARENT_SCOPE)
