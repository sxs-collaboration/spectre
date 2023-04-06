# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Burgers
  BurgersAnalyticData
  BurgersSolutions
  DgSubcell
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FiniteDifference
  Informer
  Limiters
  LinearOperators
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  )

add_spectre_parallel_executable(
  "EvolveBurgers"
  EvolveBurgers
  Evolution/Executables/Burgers
  "EvolutionMetavars"
  "${LIBS_TO_LINK}"
  )
