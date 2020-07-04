# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearSolver)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Gmres.cpp
  Lapack.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Gmres.hpp
  InnerProduct.hpp
  Lapack.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  Convergence
  DataStructures
  Options
  PRIVATE
  Lapack
  )
