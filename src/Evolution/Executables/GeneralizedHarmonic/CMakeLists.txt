# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_generalized_harmonic_executable
    INITIAL_DATA_NAME INITIAL_DATA BOUNDARY_CONDITIONS LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGh${INITIAL_DATA_NAME}"
    EvolveGeneralizedHarmonicWithHorizon
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable)

function(add_generalized_harmonic_no_horizon_executable
    INITIAL_DATA_NAME INITIAL_DATA BOUNDARY_CONDITIONS LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGh${INITIAL_DATA_NAME}"
    EvolveGeneralizedHarmonic
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_no_horizon_executable)

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
  PhaseControl
  Time
  Utilities
  )

add_generalized_harmonic_executable(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  "${LIBS_TO_LINK};ApparentHorizons;GeneralRelativitySolutions"
)

add_generalized_harmonic_executable(
  KerrSchildNumericInitialData
  evolution::NumericInitialData
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  "${LIBS_TO_LINK};ApparentHorizons;GeneralRelativitySolutions;Importers"
)

add_generalized_harmonic_no_horizon_executable(
  GaugeWave
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<3>>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<3>>
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_generalized_harmonic_no_horizon_executable(
  GaugeWaveNumericInitialData
  evolution::NumericInitialData
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<3>>
  "${LIBS_TO_LINK};GeneralRelativitySolutions;Importers"
)

