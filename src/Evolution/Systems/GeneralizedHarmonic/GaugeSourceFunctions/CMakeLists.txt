# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralizedHarmonicGaugeSourceFunctions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DampedHarmonic.cpp
  DampedWaveHelpers.cpp
  DhGaugeParameters.cpp
  Gauges.cpp
  Harmonic.cpp
  InitializeDampedHarmonic.cpp
  RegisterDerived.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DampedHarmonic.hpp
  DampedWaveHelpers.hpp
  DhGaugeParameters.hpp
  Factory.hpp
  Gauges.hpp
  Harmonic.hpp
  InitializeDampedHarmonic.hpp
  RegisterDerived.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  DataStructures
  Domain
  ErrorHandling
  FunctionsOfTime
  GeneralizedHarmonic
  GeneralRelativity
  Initialization
  Options
  Spectral
  Utilities
  )

add_subdirectory(Tags)
