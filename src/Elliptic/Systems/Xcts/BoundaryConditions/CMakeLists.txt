# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Flatness.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Flatness.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Elliptic
  Options
  Parallel
  Utilities
  )
