# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE SolveXcts)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SolveXcts.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  AmrCriteria
  Charmxx::main
  Convergence
  CoordinateMaps
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainCreators
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  EllipticSubdomainPreconditioners
  ErrorHandling
  Events
  EventsAndTriggers
  FunctionsOfTime
  Hydro
  Informer
  Initialization
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
  Utilities
  Xcts
  XctsAnalyticData
  XctsBoundaryConditions
  XctsEvents
  XctsSolutions
  )
