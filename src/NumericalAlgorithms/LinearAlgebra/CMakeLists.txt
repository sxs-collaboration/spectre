# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearAlgebra)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FindGeneralizedEigenvalues.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FindGeneralizedEigenvalues.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  DataStructures
  ErrorHandling
  PRIVATE
  Lapack
  )
