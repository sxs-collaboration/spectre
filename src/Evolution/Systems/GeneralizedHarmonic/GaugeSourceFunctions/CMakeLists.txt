# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DampedHarmonic.cpp
  DampedWaveHelpers.cpp
  DhGaugeParameters.cpp
  InitializeDampedHarmonic.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DampedHarmonic.hpp
  DampedWaveHelpers.hpp
  DhGaugeParameters.hpp
  Gauges.hpp
  InitializeDampedHarmonic.hpp
  )

add_subdirectory(Tags)
