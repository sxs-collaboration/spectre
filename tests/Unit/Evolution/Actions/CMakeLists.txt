# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionActions")

set(LIBRARY_SOURCES
  Test_ComputeVolumeDuDt.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Actions/"
  "${LIBRARY_SOURCES}"
  "" # Header-only, link dependencies included from testing lib
  )
