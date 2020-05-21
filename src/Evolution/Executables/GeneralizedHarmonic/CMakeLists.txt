# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ApparentHorizons
  CoordinateMaps
  DiscontinuousGalerkin
  Domain
  DomainCreators
  Evolution
  GeneralRelativity
  GeneralizedHarmonic
  GeneralizedHarmonicGaugeSourceFunctions
  IO
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Options
  Time
  Utilities
  )

function(add_generalized_harmonic_executable
  EXECUTABLE_NAME INITIAL_DATA BOUNDARY_CONDITIONS)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}"
    EvolveGeneralizedHarmonic
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable)

add_generalized_harmonic_executable(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
)
target_link_libraries(
  EvolveKerrSchild
  PRIVATE
  GeneralRelativitySolutions
  )

add_generalized_harmonic_executable(
  KerrSchildNumericInitialData
  evolution::NumericInitialData<GeneralizedHarmonic::System<3>>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
)
target_link_libraries(
  EvolveKerrSchildNumericInitialData
  PRIVATE
  GeneralRelativitySolutions
  )
