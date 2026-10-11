# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "SolveBnsInitialData")

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SolveBnsInitialData.cpp
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
  BnsInitialData
  BnsInitialDataAnalyticData
  BnsInitialDataBoundaryConditions
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  PhaseControl
  RelativisticEulerSolutions
  Utilities
)



target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
