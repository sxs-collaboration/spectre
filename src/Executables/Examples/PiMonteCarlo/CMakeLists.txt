# Distributed under the MIT License.
# See LICENSE.txt for details.

foreach(EXECUTABLE PiMonteCarlo PiMonteCarloStarter)
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    ${EXECUTABLE}.cpp
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
endforeach()
