# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE EvolveBurgers)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveBurgers.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Burgers
  BurgersAnalyticData
  BurgersSolutions
  Charmxx::main
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
