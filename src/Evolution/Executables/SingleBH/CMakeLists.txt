# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  GeneralizedHarmonic
  GeneralRelativitySolutions
  IO
  Informer
  LinearOperators
  MathFunctions
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveSingleBH
  EvolveSingleBH
  Evolution/Executables/SingleBH
  EvolutionMetavars
  "${LIBS_TO_LINK}"
)
