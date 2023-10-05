# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
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
  ScalarAdvection
  ScalarAdvectionSolutions
  Serialization
  Time
  Utilities
  )

function(add_scalar_advection_executable DIM)
  set(EXECUTABLE "EvolveScalarAdvection${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveScalarAdvection.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_scalar_advection_executable(1)
add_scalar_advection_executable(2)
