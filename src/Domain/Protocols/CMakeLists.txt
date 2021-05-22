# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Domain)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Metavariables.hpp
  )
