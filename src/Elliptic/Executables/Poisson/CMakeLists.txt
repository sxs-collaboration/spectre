# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Charmxx::main
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  Events
  EventsAndTriggers
  FunctionsOfTime
  Informer
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  Poisson
  PoissonBoundaryConditions
  PoissonSolutions
  Utilities
  )

function(add_poisson_executable DIM)
  set(EXECUTABLE "SolvePoisson${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    SolvePoisson.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction(add_poisson_executable)

add_poisson_executable(1)
add_poisson_executable(2)
add_poisson_executable(3)
