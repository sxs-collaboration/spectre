# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NumericalDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_LiftFlux.cpp
  )

add_subdirectory(Actions)

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "" # Header-only, link dependencies included from testing lib
  )
