# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Burgers
  BurgersSolutions
  DiscontinuousGalerkin
  DomainCreators
  IO
  Informer
  LinearOperators
  SlopeLimiters
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveBurgers
  EvolveBurgers
  Evolution/Executables/Burgers
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
