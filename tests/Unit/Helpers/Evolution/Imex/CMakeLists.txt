# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "ImexHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  TestSector.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  Imex
  Utilities
  )
