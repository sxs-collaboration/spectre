# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CurvedWaveEquationAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  PureSphericalHarmonic.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  PureSphericalHarmonic.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  AnalyticData
  CurvedScalarWave
  Options
  ScalarWave
  PUBLIC
  DataStructures
  GeneralRelativity
  GeneralRelativitySolutions
  Spectral
  Utilities
  WaveEquationSolutions
  )
