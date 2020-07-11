# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RadiationTransport)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  )

add_subdirectory(M1Grey)
