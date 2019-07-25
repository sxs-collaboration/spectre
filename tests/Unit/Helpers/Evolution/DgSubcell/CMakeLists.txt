# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcellHelpers)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit/
  HEADERS
  ProjectionTestHelpers.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ProjectionTestHelpers.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Spectral

  PRIVATE
  Utilities
  )
