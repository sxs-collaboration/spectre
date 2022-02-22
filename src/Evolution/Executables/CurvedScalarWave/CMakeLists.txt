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

set(INTERPOLATION_LIBS_TO_LINK
  ApparentHorizons
  ParallelInterpolation
)

function(add_curved_scalar_wave_executable EXECUTABLE_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveCurvedScalarWave${EXECUTABLE_NAME}${DIM}D"
    EvolveCurvedScalarWave
    Evolution/Executables/CurvedScalarWave
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_curved_scalar_wave_executable)

function(add_interpolating_curved_scalar_wave_executable EXECUTABLE_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveCurvedScalarWave${EXECUTABLE_NAME}${DIM}D"
    EvolveCurvedScalarWave
    Evolution/Executables/CurvedScalarWave
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK};${INTERPOLATION_LIBS_TO_LINK}"
    )
endfunction(add_interpolating_curved_scalar_wave_executable)

function(add_flat_plane_wave_executable DIM)
  add_curved_scalar_wave_executable(
    PlaneWaveMinkowski
    ${DIM}
    CurvedScalarWave::AnalyticData::ScalarWaveGr<ScalarWave::Solutions::PlaneWave<${DIM}>,gr::Solutions::Minkowski<${DIM}>>
    )
endfunction(add_flat_plane_wave_executable)

add_flat_plane_wave_executable(1)
add_flat_plane_wave_executable(2)

add_interpolating_curved_scalar_wave_executable(
  PlaneWaveMinkowski
  3
  CurvedScalarWave::AnalyticData::ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,gr::Solutions::Minkowski<3>>
  )

add_interpolating_curved_scalar_wave_executable(
  ScalarWaveKerrSchild
  3
  CurvedScalarWave::AnalyticData::ScalarWaveGr<CurvedScalarWave::AnalyticData::PureSphericalHarmonic,gr::Solutions::KerrSchild>
  )
