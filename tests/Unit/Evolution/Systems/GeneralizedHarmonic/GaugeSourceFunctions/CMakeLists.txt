# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralizedHarmonicGaugeSourceFunctions")

set(LIBRARY_SOURCES
  Test_DampedHarmonic.cpp
  Test_DampedWaveHelpers.cpp
  Test_InitializeDampedHarmonic.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  FunctionsOfTime
  GeneralRelativity
  GeneralizedHarmonic
  GeneralizedHarmonicGaugeSourceFunctions
  LinearOperators
  Options
  Parallel
  Spectral
  Time
  Utilities
  )
