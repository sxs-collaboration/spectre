# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ApparentHorizonFinder
  Charmxx::main
  CoordinateMaps
  DomainCreators
  EllipticDg
  GeneralRelativity
  GrSurfaces
  Informer
  Importers
  LinearOperators
  Observer
  Options
  Parallel
  ParallelInterpolation
  Utilities
  )

function(add_find_horizons_executable DIM TWO_HORIZONS)
  if(TWO_HORIZONS STREQUAL "true")
    set(EXECUTABLE "FindTwoHorizons${DIM}D")
  else()
    set(EXECUTABLE "FindHorizons${DIM}D")
  endif()
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    FindHorizons.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    TWO_HORIZONS=${TWO_HORIZONS}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_find_horizons_executable(3 false)
add_find_horizons_executable(3 true)
