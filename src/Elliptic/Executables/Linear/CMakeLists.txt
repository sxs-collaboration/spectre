# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Informer
  IO
  LinearOperators
  MathFunctions
  Poisson
  PoissonSolutions
  Utilities
  ${SPECTRE_LIBRARIES}
  )

add_spectre_parallel_executable(
  SolvePoissonProblem1D
  SolveLinearEllipticProblem
  Elliptic/Executables/Linear
  Metavariables<1>
  "${LIBS_TO_LINK}"
  )
add_spectre_parallel_executable(
  SolvePoissonProblem2D
  SolveLinearEllipticProblem
  Elliptic/Executables/Linear
  Metavariables<2>
  "${LIBS_TO_LINK}"
  )
add_spectre_parallel_executable(
  SolvePoissonProblem3D
  SolveLinearEllipticProblem
  Elliptic/Executables/Linear
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
