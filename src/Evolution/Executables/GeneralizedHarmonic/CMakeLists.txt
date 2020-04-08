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
    GeneralRelativitySolutions
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

add_spectre_parallel_executable(
  EvolveGeneralizedHarmonic
  EvolveGeneralizedHarmonic
  Evolution/Executables/GeneralizedHarmonic
  EvolutionMetavars
  "${LIBS_TO_LINK}"
)
