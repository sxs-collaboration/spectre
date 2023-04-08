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
  Informer
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  WaveEquationSolutions
)

set(INTERPOLATION_LIBS_TO_LINK
  ApparentHorizons
  ParallelInterpolation
)

function(add_curved_scalar_wave_executable EXECUTABLE_NAME DIM BACKGROUND INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveCurvedScalarWave${EXECUTABLE_NAME}${DIM}D"
    EvolveCurvedScalarWave
    Evolution/Executables/CurvedScalarWave
    "EvolutionMetavars<${DIM},${BACKGROUND},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_curved_scalar_wave_executable)

function(add_interpolating_curved_scalar_wave_executable EXECUTABLE_NAME DIM BACKGROUND INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveCurvedScalarWave${EXECUTABLE_NAME}${DIM}D"
    EvolveCurvedScalarWave
    Evolution/Executables/CurvedScalarWave
    "EvolutionMetavars<${DIM},${BACKGROUND},${INITIAL_DATA}>"
    "${LIBS_TO_LINK};${INTERPOLATION_LIBS_TO_LINK}"
  )
endfunction(add_interpolating_curved_scalar_wave_executable)

function(add_flat_plane_wave_executable DIM)
  add_curved_scalar_wave_executable(
    PlaneWaveMinkowski
    ${DIM}
    gr::Solutions::Minkowski<${DIM}>
    ScalarWave::Solutions::PlaneWave<${DIM}>
  )
endfunction(add_flat_plane_wave_executable)

add_flat_plane_wave_executable(1)
add_flat_plane_wave_executable(2)

add_interpolating_curved_scalar_wave_executable(
  PlaneWaveMinkowski
  3
  gr::Solutions::Minkowski<3>
  ScalarWave::Solutions::PlaneWave<3>
)

add_interpolating_curved_scalar_wave_executable(
  ScalarWaveKerrSchild
  3
  gr::Solutions::KerrSchild
  CurvedScalarWave::AnalyticData::PureSphericalHarmonic
)

add_spectre_parallel_executable(
  "EvolveWorldtubeCurvedScalarWaveKerrSchild3D"
  EvolveWorldtubeCurvedScalarWave
  Evolution/Executables/CurvedScalarWave
  "EvolutionMetavars<gr::Solutions::KerrSchild,ScalarWave::Solutions::PlaneWave<3>>"
  "${LIBS_TO_LINK};${INTERPOLATION_LIBS_TO_LINK};ScalarWaveWorldtube"
)
