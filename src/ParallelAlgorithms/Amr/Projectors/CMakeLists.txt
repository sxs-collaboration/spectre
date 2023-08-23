# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY AmrProjectors)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Mesh.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DefaultInitialize.hpp
  Mesh.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  DataStructures
  Spectral
  )
