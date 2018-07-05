# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Burgers
  BurgersSolutions
  DomainCreators
  Informer
  Time
  Utilities
  )

add_spectre_executable(
  EvolveBurgers
  Evolution/Executables/Burgers
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
