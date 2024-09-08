# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE SolvePunctures)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SolvePunctures.cpp
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
  Informer
  InitialDataUtilities
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
  PhaseControl
  Punctures
  PuncturesAmrCriteria
  PuncturesAnalyticData
  PuncturesBoundaryConditions
  PuncturesPointwiseFunctions
  PuncturesSolutions
  Utilities
  )
