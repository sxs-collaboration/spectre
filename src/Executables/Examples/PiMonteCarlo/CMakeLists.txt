# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE PiMonteCarlo)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  PiMonteCarlo.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Charmxx::main
  DataStructures
  ErrorHandling
  Informer
  Options
  Parallel
  Printf
  Utilities
  )
