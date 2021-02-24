# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DgSubcell.hpp
  Matrices.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Matrices.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Spectral

  PRIVATE
  Blas
  ErrorHandling
  Utilities
  )
