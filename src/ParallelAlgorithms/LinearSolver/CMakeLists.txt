# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelLinearSolver)

add_spectre_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Blas
  Convergence
  DataStructures
  IO
  Utilities
  )
