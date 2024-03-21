# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "SolveIrrotationalBns")

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SolveIrrotationalBns.cpp
)

set(LIBS_TO_LINK
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
  Informer
  IrrotationalBns
  IrrotationalBnsAnalyticData
  IrrotationalBnsBoundaryConditions
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  RelativisticEulerSolutions
  Utilities
)



target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
