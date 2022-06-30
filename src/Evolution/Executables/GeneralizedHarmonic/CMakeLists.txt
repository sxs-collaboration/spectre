# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_generalized_harmonic_executable_with_horizon
    INITIAL_DATA_NAME INITIAL_DATA BOUNDARY_CONDITIONS LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGh${INITIAL_DATA_NAME}"
    EvolveGeneralizedHarmonicWithHorizon
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<3, ${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable_with_horizon)

function(add_generalized_harmonic_executable_without_horizon
    INITIAL_DATA_NAME DIM INITIAL_DATA BOUNDARY_CONDITIONS LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGh${INITIAL_DATA_NAME}${DIM}D"
    EvolveGeneralizedHarmonicWithoutHorizon
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${DIM}, ${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
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
  GeneralizedHarmonicGaugeSourceFunctions
  GhConstraintDamping
  IO
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  Time
  Utilities
  )

add_generalized_harmonic_executable_with_horizon(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  "${LIBS_TO_LINK};ApparentHorizons;GeneralRelativitySolutions"
)

add_generalized_harmonic_executable_with_horizon(
  KerrSchildNumericInitialData
  evolution::NumericInitialData
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  "${LIBS_TO_LINK};ApparentHorizons;GeneralRelativitySolutions;Importers"
)

add_generalized_harmonic_executable_without_horizon(
  GaugeWave
  1
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<1>>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<1>>
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_generalized_harmonic_executable_without_horizon(
  GaugeWave
  2
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<2>>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<2>>
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_generalized_harmonic_executable_without_horizon(
  GaugeWave
  3
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<3>>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<3>>
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_spectre_parallel_executable(
  "EvolveGhBinaryBlackHole"
  EvolveGhBinaryBlackHole
  Evolution/Executables/GeneralizedHarmonic
  "EvolutionMetavars"
  "${LIBS_TO_LINK};ControlSystem;GeneralRelativitySolutions;Importers"
)
