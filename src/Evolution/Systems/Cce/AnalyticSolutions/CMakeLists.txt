# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CceAnalyticSolutions)

set(LIBRARY_SOURCES
  BouncingBlackHole.cpp
  SphericalMetricData.cpp
  WorldtubeData.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

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

