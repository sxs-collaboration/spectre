# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Charmxx::main
  CoordinateMaps
  DgSubcell
  Domain
  DomainCreators
  EventsAndTriggers
  Evolution
  Informer
  LinearOperators
  Logging
  Observer
  Options
  Parallel
  ParallelAmr
  PhaseControl
  Spectral
  Time
  Utilities
  )

function(add_export_coordinates_executable DIM)
  set(EXECUTABLE "ExportCoordinates${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    ExportCoordinates.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_export_coordinates_executable(1)
add_export_coordinates_executable(2)
add_export_coordinates_executable(3)
