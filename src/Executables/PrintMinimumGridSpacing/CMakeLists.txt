# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DomainCreators
  Informer
  IO
  Utilities
  )

add_spectre_parallel_executable(
  PrintMinimumGridSpacing1D
  PrintMinimumGridSpacing
  Executables/PrintMinimumGridSpacing
  Metavariables<1>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  PrintMinimumGridSpacing2D
  PrintMinimumGridSpacing
  Executables/PrintMinimumGridSpacing
  Metavariables<2>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  PrintMinimumGridSpacing3D
  PrintMinimumGridSpacing
  Executables/PrintMinimumGridSpacing
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
