# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AnalyticChristoffel.cpp
  DampedHarmonic.cpp
  DampedWaveHelpers.cpp
  DhGaugeParameters.cpp
  Gauges.cpp
  HalfPiPhiTwoNormals.cpp
  Harmonic.cpp
  InitializeDampedHarmonic.cpp
  RegisterDerived.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticChristoffel.hpp
  DampedHarmonic.hpp
  DampedWaveHelpers.hpp
  DhGaugeParameters.hpp
  Factory.hpp
  Gauges.hpp
  HalfPiPhiTwoNormals.hpp
  Harmonic.hpp
  InitializeDampedHarmonic.hpp
  RegisterDerived.hpp
  )

add_subdirectory(Tags)
