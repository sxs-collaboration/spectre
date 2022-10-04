# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralizedHarmonicGaugeSourceFunctions")

set(LIBRARY_SOURCES
  Test_AnalyticChristoffel.cpp
  Test_DampedHarmonic.cpp
  Test_DampedWaveHelpers.cpp
  Test_HalfPiPhiTwoNormals.cpp
  Test_Harmonic.cpp
  Test_SetPiFromGauge.cpp
  Test_Tags.cpp
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
  LinearOperators
  Options
  Parallel
  Spectral
  Time
  Utilities
  )
