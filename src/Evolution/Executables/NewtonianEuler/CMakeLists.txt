# Distributed under the MIT License.
# See LICENSE.txt for details.

# We can't do the instantiations for NewtonianEuler in the NewtonianEuler
# library because the TimeDerivativeTerms and system depend on the analytic
# solution/data being used, which causes a cyclic dependency since the solutions
# depend on the system for the tags. This cyclic dependency is already present
# in the code, but currently only through header files. The same cyclic
# dependency exists between NewtonianEulerSources and NewtonianEuler. To resolve
# these, we will need to factor the NewtonianEuler tags into a separate library.
# fter the NewtonianEuler tags are factored in this way, the instantiation
# source file should be moved to the NewtonianEuler library.
set(LIBRARY NewtonianEulerInstantiations)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  VolumeTermsInstantiation.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Evolution
  NewtonianEuler
  NewtonianEulerAnalyticData
  NewtonianEulerSolutions
  NewtonianEulerSources
  Utilities
  )

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  Hydro
  IO
  Informer
  Limiters
  LinearOperators
  MathFunctions
  NewtonianEuler
  NewtonianEulerAnalyticData
  NewtonianEulerInstantiations
  NewtonianEulerLimiters
  NewtonianEulerNumericalFluxes
  NewtonianEulerSolutions
  NewtonianEulerSources
  Options
  Parallel
  PhaseControl
  Time
  Utilities
  )

function(add_newtonian_euler_executable INITIAL_DATA_NAME DIM INITIAL_DATA)
  add_spectre_parallel_executable(
    "EvolveNewtonianEuler${INITIAL_DATA_NAME}${DIM}D"
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

function(add_kh_instability_executable DIM)
  add_newtonian_euler_executable(
    KhInstability
    ${DIM}
    NewtonianEuler::AnalyticData::KhInstability<${DIM}>
    )
endfunction(add_kh_instability_executable)

add_kh_instability_executable(2)
add_kh_instability_executable(3)

function(add_smooth_flow_executable DIM)
  add_newtonian_euler_executable(
    SmoothFlow
    ${DIM}
    NewtonianEuler::Solutions::SmoothFlow<${DIM}>
    )
endfunction(add_smooth_flow_executable)

add_smooth_flow_executable(1)
add_smooth_flow_executable(2)
add_smooth_flow_executable(3)
