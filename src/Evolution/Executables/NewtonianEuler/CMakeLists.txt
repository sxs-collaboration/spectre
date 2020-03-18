# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Hydro
  IO
  Informer
  Limiters
  LinearOperators
  MathFunctions
  NewtonianEuler
  NewtonianEulerNumericalFluxes
  NewtonianEulerSolutions
  NewtonianEulerSources
  Options
  Time
  Utilities
  )

function(add_newtonian_euler_executable EXECUTABLE_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}${DIM}D"
    EvolveNewtonianEuler
    Evolution/Executables/NewtonianEuler
    "EvolutionMetavars<${DIM},${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_newtonian_euler_executable)

function(add_riemann_problem_executable DIM)
  add_newtonian_euler_executable(
    RiemannProblem
    ${DIM}
    NewtonianEuler::Solutions::RiemannProblem<${DIM}>
    )
endfunction(add_riemann_problem_executable)

add_riemann_problem_executable(1)
add_riemann_problem_executable(2)
add_riemann_problem_executable(3)

function(add_isentropic_vortex_executable DIM)
  add_newtonian_euler_executable(
    IsentropicVortex
    ${DIM}
    NewtonianEuler::Solutions::IsentropicVortex<${DIM}>
    )
endfunction(add_isentropic_vortex_executable)

add_isentropic_vortex_executable(2)
add_isentropic_vortex_executable(3)

function(add_lane_emden_executable)
  add_newtonian_euler_executable(
    LaneEmdenStar
    3
    NewtonianEuler::Solutions::LaneEmdenStar
    )
endfunction(add_lane_emden_executable)

add_lane_emden_executable()
