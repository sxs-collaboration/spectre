# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Amr
  AmrCriteria
  AmrProjectors
  CoordinateMaps
  DgSubcell
  Domain
  DomainCreators
  EventsAndTriggers
  Evolution
  Informer
  LinearOperators
  Observer
  Options
  Parallel
  PhaseControl
  Spectral
  Time
  Utilities
  )

add_spectre_parallel_executable(
  ExportCoordinates1D
  ExportCoordinates
  Executables/ExportCoordinates
  Metavariables<1,false>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  ExportCoordinates2D
  ExportCoordinates
  Executables/ExportCoordinates
  Metavariables<2,false>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  ExportCoordinates3D
  ExportCoordinates
  Executables/ExportCoordinates
  Metavariables<3,false>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  ExportTimeDependentCoordinates3D
  ExportCoordinates
  Executables/ExportCoordinates
  Metavariables<3,true>
  "${LIBS_TO_LINK}"
  )
