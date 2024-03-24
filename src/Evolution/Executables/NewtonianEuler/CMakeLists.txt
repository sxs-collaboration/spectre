# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Actions
  Charmxx::main
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FiniteDifference
  Hydro
  Informer
  Limiters
  LinearOperators
  MathFunctions
  NewtonianEuler
  NewtonianEulerAnalyticData
  NewtonianEulerLimiters
  NewtonianEulerSolutions
  NewtonianEulerSources
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  )

function(add_newtonian_euler_executable DIM)
  set(EXECUTABLE "EvolveNewtonianEuler${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveNewtonianEuler.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_newtonian_euler_executable(1)
add_newtonian_euler_executable(2)
add_newtonian_euler_executable(3)
