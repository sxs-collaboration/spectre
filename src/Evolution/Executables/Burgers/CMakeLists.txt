# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Burgers
  BurgersAnalyticData
  BurgersSolutions
  DiscontinuousGalerkin
  DomainCreators
  Evolution
  IO
  Informer
  Limiters
  LinearOperators
  Options
  Time
  Utilities
  )

function(add_burgers_executable EXECUTABLE_NAME INITIAL_DATA)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}"
    EvolveBurgers
    Evolution/Executables/Burgers
    "EvolutionMetavars<${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_burgers_executable)

add_burgers_executable(
  Step
  Burgers::Solutions::Step
  )
