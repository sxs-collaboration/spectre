# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActiveGrid.hpp
  DgSubcell.hpp
  Matrices.hpp
  Mesh.hpp
  Projection.hpp
  Reconstruction.hpp
  TwoMeshRdmpTci.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ActiveGrid.cpp
  Matrices.cpp
  Mesh.cpp
  Projection.cpp
  Reconstruction.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  Spectral

  PRIVATE
  Blas
  ErrorHandling
  Utilities
  )

add_subdirectory(Tags)
