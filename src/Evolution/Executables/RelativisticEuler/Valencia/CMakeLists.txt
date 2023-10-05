# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
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
  Informer
  Limiters
  LinearOperators
  MathFunctions
  RelativisticEulerSolutions
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  Valencia
  )

function(add_valencia_executable SUFFIX DIM INITIAL_DATA)
  set(EXECUTABLE "EvolveValencia${SUFFIX}${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveValencia.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    INITIAL_DATA=${INITIAL_DATA}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

function(add_smooth_flow_executable DIM)
  add_valencia_executable(
    SmoothFlow
    ${DIM}
    RelativisticEuler::Solutions::SmoothFlow<${DIM}>
    )
endfunction(add_smooth_flow_executable)

add_smooth_flow_executable(1)
add_smooth_flow_executable(2)
add_smooth_flow_executable(3)
