# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActiveGrid.hpp
  Inactive.hpp
  Mesh.hpp
  TciGridHistory.hpp
  )
