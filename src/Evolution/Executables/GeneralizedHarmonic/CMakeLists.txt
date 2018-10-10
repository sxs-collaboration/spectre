# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    CoordinateMaps
    DiscontinuousGalerkin
    Domain
    DomainCreators
    GeneralRelativity
    GeneralRelativitySolutions
    IO
    Informer
    LinearOperators
    MathFunctions
    Time
    Utilities
    ${SPECTRE_LIBRARIES}
  )

add_spectre_executable(
  EvolveKerrSchild
  EvolveKerrSchild
  Evolution/Executables/GeneralizedHarmonic
  EvolutionMetavars
  "${LIBS_TO_LINK}"
)
