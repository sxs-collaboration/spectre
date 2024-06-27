# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE EvolveM1GreyConstantM1)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveM1Grey.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Actions
  Charmxx::main
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Events
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativitySolutions
  Hydro
  Imex
  Informer
  Limiters
  LinearOperators
  MathFunctions
  M1Grey
  M1GreySolutions
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  )

