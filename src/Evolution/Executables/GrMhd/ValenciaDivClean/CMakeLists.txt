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
  RelativisticEulerSolutions
  SlopeLimiters
  Time
  Utilities
  ValenciaDivClean
  )

add_spectre_parallel_executable(
  EvolveValenciaDivClean
  EvolveValenciaDivClean
  Evolution/Executables/GrMhd/ValenciaDivClean
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  GeneralRelativitySolutions
  GrMhdAnalyticData
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

add_spectre_parallel_executable(
  EvolveValenciaDivCleanAnalyticData
  EvolveValenciaDivCleanAnalyticData
  Evolution/Executables/GrMhd/ValenciaDivClean
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
