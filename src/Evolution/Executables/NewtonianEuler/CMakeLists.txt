# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Hydro
  IO
  Informer
  Limiters
  LinearOperators
  MathFunctions
  NewtonianEuler
  NewtonianEulerSolutions
  NewtonianEulerSources
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveNewtonianEuler1D
  EvolveNewtonianEuler
  Evolution/Executables/NewtonianEuler
  EvolutionMetavars<1>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  EvolveNewtonianEuler2D
  EvolveNewtonianEuler
  Evolution/Executables/NewtonianEuler
  EvolutionMetavars<2>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  EvolveNewtonianEuler3D
  EvolveNewtonianEuler
  Evolution/Executables/NewtonianEuler
  EvolutionMetavars<3>
  "${LIBS_TO_LINK}"
  )
