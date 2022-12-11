# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_generalized_harmonic_executable_without_horizon
    DIM LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGhNoBlackHole${DIM}D"
    EvolveGeneralizedHarmonicNoBlackHole
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${DIM}, false>"
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
  GeneralizedHarmonic
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveGhSingleBlackHole
  EvolveGeneralizedHarmonicSingleBlackHole
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars<3, false>"
  "${LIBS_TO_LINK};ApparentHorizons;GeneralRelativitySolutions"
  )

add_spectre_parallel_executable(
  EvolveGhSingleBlackHoleNumericalInitialData
  EvolveGeneralizedHarmonicSingleBlackHole
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars<3, true>"
  "${LIBS_TO_LINK};ApparentHorizons;GeneralRelativitySolutions;Importers"
  )

add_generalized_harmonic_executable_without_horizon(
  1
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_generalized_harmonic_executable_without_horizon(
  2
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_generalized_harmonic_executable_without_horizon(
  3
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_spectre_parallel_executable(
  "EvolveGhBinaryBlackHole"
  EvolveGhBinaryBlackHole
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars"
  "${LIBS_TO_LINK};ControlSystem;GeneralRelativitySolutions;Importers"
)

add_spectre_parallel_executable(
  "EvolveGhBbhCce"
  EvolveGhBbhCce
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars"
  "${LIBS_TO_LINK};Cce;ControlSystem;GeneralRelativitySolutions;Importers"
)
