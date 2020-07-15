# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CceAnalyticSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BouncingBlackHole.cpp
  LinearizedBondiSachs.cpp
  SphericalMetricData.cpp
  TeukolskyWave.cpp
  WorldtubeData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BouncingBlackHole.hpp
  LinearizedBondiSachs.hpp
  SphericalMetricData.hpp
  WorldtubeData.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralizedHarmonic
  GeneralRelativity
  IO
  Options
  Spectral
  Utilities
  )

