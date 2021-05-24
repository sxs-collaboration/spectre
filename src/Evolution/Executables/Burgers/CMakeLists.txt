# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Burgers
  BurgersAnalyticData
  BurgersSolutions
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
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

function(add_burgers_executable INITIAL_DATA_NAME INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveBurgers${INITIAL_DATA_NAME}"
    EvolveBurgers
    Evolution/Executables/Burgers
    "EvolutionMetavars<${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_burgers_executable)

add_burgers_executable(
  Bump
  Burgers::Solutions::Bump
  )

add_burgers_executable(
  Linear
  Burgers::Solutions::Linear
  )

add_burgers_executable(
  Step
  Burgers::Solutions::Step
  )

add_burgers_executable(
  Sinusoid
  Burgers::AnalyticData::Sinusoid
  )
