# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
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
  add_spectre_parallel_executable(
    "EvolveScalarWave${DIM}D"
    EvolveScalarWave
    Evolution/Executables/ScalarWave
    "EvolutionMetavars<${DIM}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_scalar_wave_executable)

add_scalar_wave_executable(1)
add_scalar_wave_executable(2)
add_scalar_wave_executable(3)
