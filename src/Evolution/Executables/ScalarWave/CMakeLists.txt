# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Evolution
  IO
  Informer
  LinearOperators
  MathFunctions
  ScalarWave
  Time
  Options
  Utilities
  WaveEquationSolutions
  )

function(add_scalar_wave_executable INITIAL_DATA_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveScalarWave${INITIAL_DATA_NAME}${DIM}D"
    EvolveScalarWave
    Evolution/Executables/ScalarWave
    "EvolutionMetavars<${DIM}, ${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_scalar_wave_executable)

function(add_plane_wave_executable DIM)
  add_scalar_wave_executable(
    PlaneWave
    ${DIM}
    ScalarWave::Solutions::PlaneWave<${DIM}>
    )
endfunction(add_plane_wave_executable)

add_plane_wave_executable(1)
add_plane_wave_executable(2)
add_plane_wave_executable(3)

add_scalar_wave_executable(
  RegularSphericalWave
  3
  ScalarWave::Solutions::RegularSphericalWave
  )
