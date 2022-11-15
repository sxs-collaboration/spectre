# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ConstitutiveRelations
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elasticity
  ElasticityBoundaryConditions
  ElasticityPointwiseFunctions
  ElasticitySolutions
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  EllipticSubdomainPreconditioners
  Events
  EventsAndTriggers
  Informer
  LinearOperators
  Observer
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  Utilities
  )

function(add_elasticity_executable DIM)
  add_spectre_parallel_executable(
    "SolveElasticity${DIM}D"
    SolveElasticity
    Elliptic/Executables/Elasticity
    "Metavariables<${DIM}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_elasticity_executable)

add_elasticity_executable(2)
add_elasticity_executable(3)
