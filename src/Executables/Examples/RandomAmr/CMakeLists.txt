# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Amr
  AmrCriteria
  AmrProjectors
  DomainCreators
  Informer
  Options
  PhaseControl
  Spectral
  Utilities
  )

add_spectre_parallel_executable(
  RandomAmr1D
  RandomAmr
  Executables/Examples/RandomAmr
  RandomAmrMetavars<1>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  RandomAmr2D
  RandomAmr
  Executables/Examples/RandomAmr
  RandomAmrMetavars<2>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  RandomAmr3D
  RandomAmr
  Executables/Examples/RandomAmr
  RandomAmrMetavars<3>
  "${LIBS_TO_LINK}"
  )
