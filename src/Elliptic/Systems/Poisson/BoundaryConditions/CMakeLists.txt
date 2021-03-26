# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PoissonBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Robin.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Robin.hpp
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
