# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AnalyticChristoffel.cpp
  DampedHarmonic.cpp
  DampedWaveHelpers.cpp
  DhGaugeParameters.cpp
  Dispatch.cpp
  Gauges.cpp
  HalfPiPhiTwoNormals.cpp
  Harmonic.cpp
  RegisterDerived.cpp
  SetPiFromGauge.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticChristoffel.hpp
  DampedHarmonic.hpp
  DampedWaveHelpers.hpp
  DhGaugeParameters.hpp
  Dispatch.hpp
  Factory.hpp
  Gauges.hpp
  HalfPiPhiTwoNormals.hpp
  Harmonic.hpp
  RegisterDerived.hpp
  SetPiFromGauge.hpp
  )

add_subdirectory(Tags)
