# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Amr)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Flags.hpp
  NeighborFlags.hpp
  Tags.hpp
  )
