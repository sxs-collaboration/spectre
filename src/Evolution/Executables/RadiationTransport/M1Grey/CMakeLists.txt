# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  GeneralRelativitySolutions
  Hydro
  IO
  Informer
  Limiters
  LinearOperators
  MathFunctions
  M1Grey
  M1GreySolutions
  Options
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveStandaloneM1
  EvolveStandaloneM1
  Evolution/Executables/RadiationTransport/M1Grey
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
