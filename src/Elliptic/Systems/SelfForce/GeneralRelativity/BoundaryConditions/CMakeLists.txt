# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrSelfForceBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Angular.cpp
  Sommerfeld.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Angular.hpp
  Sommerfeld.hpp
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
