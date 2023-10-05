# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Charmxx::main
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Events
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  Informer
  LinearOperators
  MathFunctions
  ScalarWave
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  WaveEquationSolutions
  )

function(add_scalar_wave_executable DIM)
  set(EXECUTABLE "EvolveScalarWave${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveScalarWave.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction(add_scalar_wave_executable)

add_scalar_wave_executable(1)
add_scalar_wave_executable(2)
add_scalar_wave_executable(3)
