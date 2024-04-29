# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Spectral)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Basis.cpp
  Chebyshev.cpp
  Filtering.cpp
  FiniteDifference.cpp
  Legendre.cpp
  LogicalCoordinates.cpp
  Mesh.cpp
  Projection.cpp
  Quadrature.cpp
  Spectral.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Basis.hpp
  Clenshaw.hpp
  Filtering.hpp
  LogicalCoordinates.hpp
  Mesh.hpp
  Projection.hpp
  Quadrature.hpp
  Spectral.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  PRIVATE
  BLAS::BLAS
  LAPACK::LAPACK
  RootFinding
  )

add_subdirectory(Python)
