# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  Events
  EventsAndTriggers
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

function(add_irrotational_bns_executable)
  add_spectre_parallel_executable(
    "SolveIrrotationalBns"
    SolveIrrotationalBns
    Elliptic/Executables/IrrotationalBns
    "Metavariables"
    "${LIBS_TO_LINK}"
    )
endfunction(add_irrotational_bns_executable)

add_irrotational_bns_executable()
