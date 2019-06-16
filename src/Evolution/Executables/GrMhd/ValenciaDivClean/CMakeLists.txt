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
  Limiters
  LinearOperators
  MathFunctions
  RelativisticEulerSolutions
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
  Limiters
  LinearOperators
  MathFunctions
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
