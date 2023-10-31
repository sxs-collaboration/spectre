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
  Initialization
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelNonlinearSolver
  ParallelSchwarz
  Utilities
  Punctures
  PuncturesAnalyticData
  PuncturesBoundaryConditions
  PuncturesPointwiseFunctions
  PuncturesSolutions
  )
