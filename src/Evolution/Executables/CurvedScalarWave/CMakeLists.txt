# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  CurvedScalarWave
  CurvedWaveEquationAnalyticData
  DiscontinuousGalerkin
  DomainCreators
  Events
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativity
  IO
  Informer
  LinearOperators
  MathFunctions
  Time
  Options
  Parallel
  PhaseControl
  Utilities
  WaveEquationSolutions
)

function(add_curved_scalar_wave_executable EXECUTABLE_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}${DIM}D"
    EvolveCurvedScalarWave
    Evolution/Executables/CurvedScalarWave
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_curved_scalar_wave_executable)

function(add_flat_plane_wave_executable DIM)
  add_curved_scalar_wave_executable(
    PlaneWaveMinkowski
    ${DIM}
    CurvedScalarWave::AnalyticData::ScalarWaveGr<ScalarWave::Solutions::PlaneWave<${DIM}>,gr::Solutions::Minkowski<${DIM}>>
    )
endfunction(add_flat_plane_wave_executable)

add_flat_plane_wave_executable(1)
add_flat_plane_wave_executable(2)
add_flat_plane_wave_executable(3)
