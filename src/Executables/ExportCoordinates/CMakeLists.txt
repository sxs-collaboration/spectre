# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Amr
  AmrCriteria
  AmrProjectors
  Charmxx::main
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

function(add_export_coordinates_executable DIM TIME_DEPENDENT)
  if(TIME_DEPENDENT STREQUAL "true")
    set(EXECUTABLE "ExportTimeDependentCoordinates${DIM}D")
  else()
    set(EXECUTABLE "ExportCoordinates${DIM}D")
  endif()
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    ExportCoordinates.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    TIME_DEPENDENT=${TIME_DEPENDENT}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_export_coordinates_executable(1 false)
add_export_coordinates_executable(2 false)
add_export_coordinates_executable(3 false)
add_export_coordinates_executable(3 true)
