# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BouncingBlackHole.cpp
  LinearizedBondiSachs.cpp
  GaugeWave.cpp
  RobinsonTrautman.cpp
  RotatingSchwarzschild.cpp
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
  GaugeWave.hpp
  RobinsonTrautman.hpp
  RotatingSchwarzschild.hpp
  SphericalMetricData.hpp
  TeukolskyWave.hpp
  WorldtubeData.hpp
  )
