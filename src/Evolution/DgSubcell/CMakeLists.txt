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
  NeighborData.hpp
  PerssonTci.hpp
  Projection.hpp
  Reconstruction.hpp
  SubcellOptions.hpp
  TciStatus.hpp
  TwoMeshRdmpTci.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ActiveGrid.cpp
  Matrices.cpp
  Mesh.cpp
  NeighborData.cpp
  PerssonTci.cpp
  Projection.cpp
  Reconstruction.cpp
  SubcellOptions.cpp
  TciStatus.cpp
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
