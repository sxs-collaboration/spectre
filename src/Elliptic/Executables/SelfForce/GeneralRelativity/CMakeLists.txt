# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "SolveGrSelfForce")

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SolveGrSelfForce.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  AmrCriteria
  Charmxx::main
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  EllipticSubdomainPreconditioners
  Events
  EventsAndTriggers
  FunctionsOfTime
  Informer
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelAmr
  ParallelLinearSolver
  ParallelMultigrid
  ParallelNonlinearSolver
  ParallelSchwarz
  PhaseControl
  GrSelfForce
  GrSelfForceAmrCriteria
  GrSelfForceAnalyticData
  GrSelfForceBoundaryConditions
  )
