# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NumericalFluxes")

set(LIBRARY_SOURCES
  Test_Hll.cpp
  Test_LocalLaxFriedrichs.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/"
  "${LIBRARY_SOURCES}"
  "" # Header-only, link dependencies included from testing lib
  )
