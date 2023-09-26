# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "EvolveForceFree")

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveForceFree.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Charmxx::main
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
