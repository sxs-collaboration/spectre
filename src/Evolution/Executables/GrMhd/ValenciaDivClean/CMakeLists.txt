# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_grmhd_executable LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveValenciaDivClean"
    EvolveValenciaDivClean
    Evolution/Executables/GrMhd/ValenciaDivClean
    "EvolutionMetavars<false>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_grmhd_executable)

set(LIBS_TO_LINK
  CoordinateMaps
  DgSubcell
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FiniteDifference
  GeneralRelativitySolutions
  GrMhdAnalyticData
  GrMhdSolutions
  Hydro
  Informer
  Limiters
  LinearOperators
  MathFunctions
  ParallelInterpolation
  RelativisticEulerSolutions
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )
  add_grmhd_executable(
  "${LIBS_TO_LINK}"
  )
