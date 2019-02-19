# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  ConstitutiveRelations
  DiscontinuousGalerkin
  DomainCreators
  Elasticity
  ElasticitySolutions
  Informer
  IO
  LinearOperators
  LinearSolver
  MathFunctions
  Utilities
  ${SPECTRE_LIBRARIES}
  )

add_spectre_parallel_executable(
  SolveElasticityProblem2D
  SolveElasticityProblem
  Elliptic/Executables/Elasticity
  Metavariables<2>
  "${LIBS_TO_LINK}"
  )
add_spectre_parallel_executable(
  SolveElasticityProblem3D
  SolveElasticityProblem
  Elliptic/Executables/Elasticity
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
