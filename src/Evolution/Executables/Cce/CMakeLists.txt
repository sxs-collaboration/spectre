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
  Parallel
  Utilities
  )

add_spectre_parallel_executable(
  CharacteristicExtract
  CharacteristicExtract
  Evolution/Executables/Cce
  EvolutionMetavars<Cce::H5WorldtubeBoundary>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  AnalyticTestCharacteristicExtract
  CharacteristicExtract
  Evolution/Executables/Cce
  EvolutionMetavars<Cce::AnalyticWorldtubeBoundary>
  "${LIBS_TO_LINK}"
  )
