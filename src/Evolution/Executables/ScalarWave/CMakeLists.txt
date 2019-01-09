# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    CoordinateMaps
    DiscontinuousGalerkin
    DomainCreators
    IO
    Informer
    LinearOperators
    MathFunctions
    ScalarWave
    Time
    Utilities
    WaveEquation
    ${SPECTRE_LIBRARIES}
  )

add_spectre_parallel_executable(
  EvolveScalarWave1D
  EvolveScalarWave
  Evolution/Executables/ScalarWave
  EvolutionMetavars<1>
  "${LIBS_TO_LINK}"
)
add_spectre_parallel_executable(
  EvolveScalarWave2D
  EvolveScalarWave
  Evolution/Executables/ScalarWave
  EvolutionMetavars<2>
  "${LIBS_TO_LINK}"
)
add_spectre_parallel_executable(
  EvolveScalarWave3D
  EvolveScalarWave
  Evolution/Executables/ScalarWave
  EvolutionMetavars<3>
  "${LIBS_TO_LINK}"
)
