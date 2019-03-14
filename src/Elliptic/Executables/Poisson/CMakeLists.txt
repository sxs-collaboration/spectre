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
  SolvePoissonProblem
  Elliptic/Executables/Poisson
  Metavariables<1>
  "${LIBS_TO_LINK}"
  )
add_spectre_parallel_executable(
  SolvePoissonProblem2D
  SolvePoissonProblem
  Elliptic/Executables/Poisson
  Metavariables<2>
  "${LIBS_TO_LINK}"
  )
add_spectre_parallel_executable(
  SolvePoissonProblem3D
  SolvePoissonProblem
  Elliptic/Executables/Poisson
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
