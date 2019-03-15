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
  ExportCoordinates1D
  ExportCoordinates
  Executables/ExportCoordinates
  Metavariables<1>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  ExportCoordinates2D
  ExportCoordinates
  Executables/ExportCoordinates
  Metavariables<2>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  ExportCoordinates3D
  ExportCoordinates
  Executables/ExportCoordinates
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
