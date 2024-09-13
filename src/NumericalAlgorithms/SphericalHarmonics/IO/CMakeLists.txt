# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SphericalHarmonicsIO)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FillYlmLegendAndData.cpp
  ReadSurfaceYlm.cpp
  StrahlkorperCoordsToTextFile.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FillYlmLegendAndData.hpp
  ReadSurfaceYlm.hpp
  StrahlkorperCoordsToTextFile.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  H5
  PUBLIC
  SphericalHarmonics
  Utilities
  )
