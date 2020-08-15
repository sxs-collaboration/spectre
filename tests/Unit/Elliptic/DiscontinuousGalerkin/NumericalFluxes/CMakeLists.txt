# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticNumericalFluxes")

set(LIBRARY_SOURCES
  Test_FirstOrderInternalPenalty.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/DiscontinuousGalerkin/NumericalFluxes"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
