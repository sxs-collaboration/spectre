# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "InterpolationHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  Examples.hpp
  InterpolateOnElementTestHelpers.hpp
  InterpolationTargetTestHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  ApparentHorizonFinder
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  GeneralRelativity
  Observer
  Options
  Parallel
  ParallelInterpolation
  Spectral
  SphericalHarmonics
  Time
  Utilities
  )
