# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE PiMonteCarlo)
set(TUTORIAL_EXECUTABLE PiMonteCarloTutorial)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  PiMonteCarlo.cpp
  )

add_spectre_executable(
  ${TUTORIAL_EXECUTABLE}
  EXCLUDE_FROM_ALL
  PiMonteCarloTutorial.cpp
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

target_link_libraries(
  ${TUTORIAL_EXECUTABLE}
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
