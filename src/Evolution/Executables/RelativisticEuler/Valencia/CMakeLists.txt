# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Evolution
  GeneralRelativitySolutions
  Hydro
  IO
  Informer
  Limiters
  LinearOperators
  MathFunctions
  RelativisticEulerSolutions
  Time
  Options
  Utilities
  Valencia
  )

function(add_valencia_executable EXECUTABLE_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}${DIM}D"
    EvolveValencia
    Evolution/Executables/RelativisticEuler/Valencia
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_valencia_executable)

function(add_smooth_flow_executable DIM)
  add_valencia_executable(
    ValenciaSmoothFlow
    ${DIM}
    RelativisticEuler::Solutions::SmoothFlow<${DIM}>
    )
endfunction(add_smooth_flow_executable)

add_smooth_flow_executable(1)
add_smooth_flow_executable(2)
add_smooth_flow_executable(3)
