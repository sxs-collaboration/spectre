# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EventsAndDenseTriggers)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DenseTrigger.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  Parallel
  Utilities
  )
