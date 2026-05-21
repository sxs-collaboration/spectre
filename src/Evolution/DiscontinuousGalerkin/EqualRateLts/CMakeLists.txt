# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  EqualRateRegionGenerator.hpp
  EqualRateRegions.hpp
  EqualRateRegions.tpp
  FixedLtsRatio.hpp
  NonconformingEqualRateRegions.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  EqualRateRegions.cpp
  FixedLtsRatio.cpp
  NonconformingEqualRateRegions.cpp
  )

add_subdirectory(Tags)
