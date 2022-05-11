# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "ObserverHelpers")

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PUBLIC
  MockH5.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  MockH5.hpp
  ObserverHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  Framework
  IO
  Parallel
  Utilities
  )
