# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_generalized_harmonic_executable_without_horizon
    DIM LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGhNoBlackHole${DIM}D"
    EvolveGhNoBlackHole
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${DIM}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable_without_horizon)

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  Domain
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativity
  GeneralRelativityAnalyticData
  GeneralRelativitySolutions
  GeneralizedHarmonic
  Importers
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  Serialization
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveGhSingleBlackHole
  EvolveGhSingleBlackHole
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars<3>"
  "${LIBS_TO_LINK};ApparentHorizons;ControlSystem"
  )

add_generalized_harmonic_executable_without_horizon(
  1
  "${LIBS_TO_LINK}"
)

add_generalized_harmonic_executable_without_horizon(
  2
  "${LIBS_TO_LINK}"
)

add_generalized_harmonic_executable_without_horizon(
  3
  "${LIBS_TO_LINK}"
)

add_spectre_parallel_executable(
  "EvolveGhBinaryBlackHole"
  EvolveGhBinaryBlackHole
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars"
  "${LIBS_TO_LINK};ApparentHorizons;Cce;ControlSystem"
)

if (TARGET SpEC::Exporter)
  target_compile_definitions(
    EvolveGhBinaryBlackHole PRIVATE HAS_SPEC_EXPORTER)
endif()

add_spectre_parallel_executable(
  "EvolveGhCce"
  EvolveGhAndCharacteristic
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars<3, false>"
  "${LIBS_TO_LINK};Cce"
)

add_spectre_parallel_executable(
  "EvolveGhCcm"
  EvolveGhAndCharacteristic
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars<3, true>"
  "${LIBS_TO_LINK};Cce"
)
