# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Cce
  CceAnalyticSolutions
  Time
  Interpolation
  Informer
  Spectral
  LinearOperators
  Options
  Utilities
  )

add_spectre_parallel_executable(
  CharacteristicExtract
  CharacteristicExtract
  Evolution/Executables/Cce
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
