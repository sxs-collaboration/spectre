# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DataBoxTestHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
  HEADERS
  TestHelpers.hpp
  TestTags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  Framework
  Utilities
  )
