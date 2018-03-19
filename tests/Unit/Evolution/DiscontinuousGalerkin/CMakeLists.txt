# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_InitializeElement.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "" # Header-only, link dependencies included from testing lib
  )
