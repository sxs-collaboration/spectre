# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RayTracer)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Exporter
  GeneralRelativity
  GeneralRelativitySolutions
  IO
  Logging
  Options
  Parallel
  Utilities
  )

add_subdirectory(BackgroundSpacetimes)
add_subdirectory(RaySources)
