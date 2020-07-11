# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralizedHarmonicGaugeSourceFunctions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DampedHarmonic.cpp
  DampedWaveHelpers.cpp
  InitializeDampedHarmonic.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DampedHarmonic.hpp
  DampedWaveHelpers.hpp
  Gauges.hpp
  InitializeDampedHarmonic.hpp
  OptionTags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralizedHarmonic
  )
