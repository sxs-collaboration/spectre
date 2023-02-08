# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  ForceFree
  ForceFreeAnalyticData
  ForceFreeSolutions
  GeneralRelativitySolutions
  IO
  Informer
  Limiters
  LinearOperators
  Options
  Parallel
  PhaseControl
  Time
  Utilities
  )

add_spectre_parallel_executable(
  "EvolveForceFree"
  EvolveForceFree
  Evolution/Executables/ForceFree
  "EvolutionMetavars"
  "${LIBS_TO_LINK}"
  )
