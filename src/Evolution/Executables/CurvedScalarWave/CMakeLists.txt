# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    CoordinateMaps
    CurvedScalarWave
    CurvedWaveEquationAnalyticData
    DiscontinuousGalerkin
    Domain
    DomainCreators
    Evolution
    GeneralRelativity
    GeneralRelativitySolutions
    GeneralizedHarmonic
    GeneralizedHarmonicGaugeSourceFunctions
    IO
    Informer
    LinearOperators
    MathFunctions
    Options
    ScalarWave
    Time
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

function(add_plane_wave_executable DIM)
  add_scalar_wave_executable(
    CurvedPlaneWave
    ${DIM}
    CurvedScalarWave::AnalyticData::ScalarWaveGr<ScalarWave::Solutions::PlaneWave<${DIM}>,gr::Solutions::KerrSchild>
    )
endfunction(add_plane_wave_executable)

add_plane_wave_executable(3)

add_scalar_wave_executable(
  CurvedRegularSphericalWave
  3
  CurvedScalarWave::AnalyticData::ScalarWaveGr<ScalarWave::Solutions::RegularSphericalWave,gr::Solutions::KerrSchild>
  )
