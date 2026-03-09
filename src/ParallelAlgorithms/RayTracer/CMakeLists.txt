# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RayTracer)

add_spectre_library(${LIBRARY})

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
