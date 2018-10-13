# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  GeneralRelativitySolutions
  GrMhdSolutions
  Hydro
  IO
  Informer
  LinearOperators
  MathFunctions
  SlopeLimiters
  Time
  Utilities
  ValenciaDivClean
  )

add_spectre_executable(
  EvolveValenciaDivClean
  EvolveValenciaDivClean
  Evolution/Executables/GrMhd/ValenciaDivClean
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
