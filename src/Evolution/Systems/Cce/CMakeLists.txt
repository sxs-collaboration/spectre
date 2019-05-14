# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Cce)

set(LIBRARY_SOURCES
  Equations.cpp
  LinearOperators.cpp
  LinearSolve.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
