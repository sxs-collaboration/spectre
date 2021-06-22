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

function(add_scalar_advection_executable INITIAL_DATA_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveScalarAdvection${INITIAL_DATA_NAME}${DIM}D"
    EvolveScalarAdvection
    Evolution/Executables/ScalarAdvection
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_scalar_advection_executable)

function(add_sinusoid_executable)
  add_scalar_advection_executable(
    Sinusoid
    1
    ScalarAdvection::Solutions::Sinusoid
    )
endfunction(add_sinusoid_executable)

add_sinusoid_executable()
