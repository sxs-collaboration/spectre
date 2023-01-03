# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Punctures)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Punctures.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  )
