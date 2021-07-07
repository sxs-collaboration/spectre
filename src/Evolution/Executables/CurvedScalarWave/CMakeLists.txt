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
  GeneralRelativitySolutions
  GeneralizedHarmonic
  GeneralizedHarmonicGaugeSourceFunctions
  IO
  Informer
  LinearOperators
  MathFunctions
  ScalarWave
  Time
  Options
  Parallel
  PhaseControl
  Utilities
  WaveEquationSolutions
)

function(add_scalar_wave_executable EXECUTABLE_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}${DIM}D"
    EvolveCurvedScalarWave
    Evolution/Executables/CurvedScalarWave
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_scalar_wave_executable)

function(add_flat_plane_wave_executable DIM)
  add_scalar_wave_executable(
    CurvedPlaneWaveMinkowski
    ${DIM}
    CurvedScalarWave::AnalyticData::ScalarWaveGr<ScalarWave::Solutions::PlaneWave<${DIM}>,gr::Solutions::Minkowski<${DIM}>>
    )
endfunction(add_flat_plane_wave_executable)

add_flat_plane_wave_executable(1)
add_flat_plane_wave_executable(2)
add_flat_plane_wave_executable(3)
