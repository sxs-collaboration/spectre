# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DiscontinuousGalerkinActions")

set(LIBRARY_SOURCES
  Test_ComputeNonconservativeBoundaryFluxes.cpp
  Test_ImposeBoundaryConditions.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/DiscontinuousGalerkin/Actions"
  "${LIBRARY_SOURCES}"
  "CoordinateMaps;DataStructures;Domain;DiscontinuousGalerkin;ErrorHandling;Spectral;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
