# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearSolver)

set(LIBRARY_SOURCES
  Lapack.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures

  PRIVATE
  Blas
  Lapack
  )
