# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
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
  ScalarAdvection
  ScalarAdvectionSolutions
  Time
  Utilities
  )

function(add_scalar_advection_executable DIM)
  add_spectre_parallel_executable(
    "EvolveScalarAdvection${DIM}D"
    EvolveScalarAdvection
    Evolution/Executables/ScalarAdvection
    "EvolutionMetavars<${DIM}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_scalar_advection_executable)

add_scalar_advection_executable(1)
add_scalar_advection_executable(2)
