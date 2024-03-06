# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  AmrCriteria
  Charmxx::main
  ConstitutiveRelations
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elasticity
  ElasticityActions
  ElasticityBoundaryConditions
  ElasticityPointwiseFunctions
  ElasticitySolutions
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  EllipticSubdomainPreconditioners
  Events
  EventsAndTriggers
  FunctionsOfTime
  Informer
  LinearOperators
  Observer
  Options
  Parallel
  ParallelAmr
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  Utilities
  )

function(add_elasticity_executable DIM)
  set(EXECUTABLE "SolveElasticity${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    SolveElasticity.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction(add_elasticity_executable)

add_elasticity_executable(2)
add_elasticity_executable(3)
