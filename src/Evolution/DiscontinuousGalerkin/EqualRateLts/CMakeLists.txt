# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  EqualRateRegionGenerator.hpp
  EqualRateRegions.hpp
  EqualRateRegions.tpp
  NonconformingEqualRateRegions.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  EqualRateRegions.cpp
  NonconformingEqualRateRegions.cpp
  )

add_subdirectory(Tags)
