# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "ObserverHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  ObserverHelpers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Domain
  Framework
  IO
  Parallel
  Utilities
  )
