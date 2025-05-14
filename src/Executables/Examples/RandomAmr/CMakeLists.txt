# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Charmxx::main
  DomainCreators
  Informer
  Logging
  Options
  ParallelAmr
  PhaseControl
  Spectral
  Utilities
  )

function(add_random_amr_executable DIM KEEP_COARSE_GRIDS)
  set(KEEP_COARSE_GRIDS_SUFFIX "")
  if (KEEP_COARSE_GRIDS)
    set(KEEP_COARSE_GRIDS_SUFFIX "KeepCoarseGrids")
  endif()
  set(EXECUTABLE "RandomAmr${KEEP_COARSE_GRIDS_SUFFIX}${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    RandomAmr.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    KEEP_COARSE_GRIDS=${KEEP_COARSE_GRIDS}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_random_amr_executable(1 false)
add_random_amr_executable(2 false)
add_random_amr_executable(3 false)
add_random_amr_executable(1 true)
add_random_amr_executable(2 true)
add_random_amr_executable(3 true)
