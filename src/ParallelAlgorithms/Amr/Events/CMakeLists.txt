# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY AmrEvents)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RefineMesh.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Events.hpp
  ObserveAmrCriteria.hpp
  RefineMesh.hpp
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Events
  Utilities
  )
