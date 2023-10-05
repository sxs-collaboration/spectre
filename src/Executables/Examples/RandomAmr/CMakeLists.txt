# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Amr
  AmrCriteria
  AmrProjectors
  Charmxx::main
  DomainCreators
  Informer
  Options
  PhaseControl
  Spectral
  Utilities
  )

function(add_random_amr_executable DIM)
  set(EXECUTABLE "RandomAmr${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    RandomAmr.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_random_amr_executable(1)
add_random_amr_executable(2)
add_random_amr_executable(3)
