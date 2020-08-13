# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Evolution
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
  EvolveM1GreyConstantM1
  EvolveM1Grey
  Evolution/Executables/RadiationTransport/M1Grey
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
