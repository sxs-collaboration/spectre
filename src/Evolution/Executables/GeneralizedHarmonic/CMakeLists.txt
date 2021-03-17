# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_generalized_harmonic_executable
    INITIAL_DATA_NAME INITIAL_DATA BOUNDARY_CONDITIONS LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGh${INITIAL_DATA_NAME}"
    EvolveGeneralizedHarmonic
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable)

set(LIBS_TO_LINK
  ApparentHorizons
  CoordinateMaps
  DiscontinuousGalerkin
  Domain
  DomainCreators
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
  Time
  Utilities
  )

add_generalized_harmonic_executable(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  "${LIBS_TO_LINK};GeneralRelativitySolutions"
)

add_generalized_harmonic_executable(
  KerrSchildNumericInitialData
  evolution::NumericInitialData
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  "${LIBS_TO_LINK};GeneralRelativitySolutions;Importers"
)
