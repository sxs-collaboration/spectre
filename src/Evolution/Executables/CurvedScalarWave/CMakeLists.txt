# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Charmxx::main
  CoordinateMaps
  CurvedScalarWave
  CurvedWaveEquationAnalyticData
  DiscontinuousGalerkin
  DomainCreators
  Events
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativity
  Informer
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  WaveEquationSolutions
)

set(INTERPOLATION_LIBS_TO_LINK
  ApparentHorizonFinder
  ParallelInterpolation
)

function(add_curved_scalar_wave_executable EXECUTABLE_SUFFIX DIM BACKGROUND INITIAL_DATA)
  set(EXECUTABLE "EvolveCurvedScalarWave${EXECUTABLE_SUFFIX}${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveCurvedScalarWave.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    BACKGROUND=${BACKGROUND}
    INITIAL_DATA=${INITIAL_DATA}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
  if (DIM EQUAL 3)
    target_link_libraries(${EXECUTABLE} PRIVATE ${INTERPOLATION_LIBS_TO_LINK})
  endif()
endfunction(add_curved_scalar_wave_executable)

function(add_flat_plane_wave_executable DIM)
  add_curved_scalar_wave_executable(
    PlaneWaveMinkowski
    ${DIM}
    gr::Solutions::Minkowski<${DIM}>
    ScalarWave::Solutions::PlaneWave<${DIM}>
  )
endfunction(add_flat_plane_wave_executable)

add_flat_plane_wave_executable(1)
add_flat_plane_wave_executable(2)
add_flat_plane_wave_executable(3)

add_curved_scalar_wave_executable(
  KerrSchild
  3
  gr::Solutions::KerrSchild
  CurvedScalarWave::AnalyticData::PureSphericalHarmonic
)

set(WORLDTUBE_EXECUTABLE EvolveWorldtubeCurvedScalarWaveKerrSchild3D)
add_spectre_executable(
  ${WORLDTUBE_EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveWorldtubeCurvedScalarWave.cpp
  )
target_link_libraries(
  ${WORLDTUBE_EXECUTABLE}
  PRIVATE
  ${LIBS_TO_LINK}
  ${INTERPOLATION_LIBS_TO_LINK}
  ScalarWaveWorldtube
  )
