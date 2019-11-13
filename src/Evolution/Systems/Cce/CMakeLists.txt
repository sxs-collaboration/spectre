# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Cce)

set(LIBRARY_SOURCES
  BoundaryData.cpp
  Equations.cpp
  InitializeCce.cpp
  LinearOperators.cpp
  LinearSolve.cpp
  PrecomputeCceDependencies.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
