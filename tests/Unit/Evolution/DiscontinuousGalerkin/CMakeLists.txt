# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EVOLUTION_DG_TESTS
  Evolution/DiscontinuousGalerkin/Test_InitializeElement.cpp
  )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${EVOLUTION_DG_TESTS}" PARENT_SCOPE)
