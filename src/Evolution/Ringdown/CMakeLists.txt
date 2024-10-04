# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Ringdown)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  StrahlkorperCoefsInRingdownDistortedFrame.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  StrahlkorperCoefsInRingdownDistortedFrame.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  DataStructures
  DomainCreators
  Domain
  H5
  Spectral
  SphericalHarmonics
  SphericalHarmonicsIO
  Utilities
  )
