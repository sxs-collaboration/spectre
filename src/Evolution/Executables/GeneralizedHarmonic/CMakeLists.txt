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
    INITIAL_DATA_NAME INITIAL_DATA BOUNDARY_CONDITIONS BJORHUS)
  add_spectre_parallel_executable(
    "EvolveGh${INITIAL_DATA_NAME}"
    EvolveGeneralizedHarmonic
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_DATA}, ${BOUNDARY_CONDITIONS}, ${BJORHUS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable)

add_generalized_harmonic_executable(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  false
)
target_link_libraries(
  EvolveGhKerrSchild
  PRIVATE
  GeneralRelativitySolutions
  )

add_generalized_harmonic_executable(
  KerrSchildCPBjorhus
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::ConstraintPreserving
  true
)
target_link_libraries(
  EvolveGhKerrSchildCPBjorhus
  PRIVATE
  GeneralRelativitySolutions
  )

add_generalized_harmonic_executable(
  KerrSchildCPPBjorhus
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::ConstraintPreservingPhysical
  true
)
target_link_libraries(
  EvolveGhKerrSchildCPPBjorhus
  PRIVATE
  GeneralRelativitySolutions
  )

add_generalized_harmonic_executable(
  KerrSchildNumericInitialData
  evolution::NumericInitialData<GeneralizedHarmonic::System<3>>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  false
)
target_link_libraries(
  EvolveGhKerrSchildNumericInitialData
  PRIVATE
  GeneralRelativitySolutions
  )

add_generalized_harmonic_executable(
  KerrSchildNumericInitialDataCPBjorhus
  evolution::NumericInitialData<GeneralizedHarmonic::System<3>>
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::ConstraintPreserving
  true
)
target_link_libraries(
  EvolveGhKerrSchildNumericInitialDataCPBjorhus
  PRIVATE
  GeneralRelativitySolutions
  )

add_generalized_harmonic_executable(
  KerrSchildNumericInitialDataCPPBjorhus
  evolution::NumericInitialData<GeneralizedHarmonic::System<3>>
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::ConstraintPreservingPhysical
  true
)
target_link_libraries(
  EvolveGhKerrSchildNumericInitialDataCPPBjorhus
  PRIVATE
  GeneralRelativitySolutions
  )
