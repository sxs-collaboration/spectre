# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
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

add_spectre_parallel_executable(
  SolvePunctures
  SolvePunctures
  Elliptic/Executables/Punctures
  "Metavariables"
  "${LIBS_TO_LINK}"
  )
