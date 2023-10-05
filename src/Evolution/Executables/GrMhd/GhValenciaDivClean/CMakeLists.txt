# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ApparentHorizonFinder
  Charmxx::main
  Cce
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativitySolutions
  GeneralizedHarmonic
  GhGrMhdAnalyticData
  GhGrMhdSolutions
  GhRelativisticEulerSolutions
  GhValenciaDivClean
  GrMhdAnalyticData
  GrMhdSolutions
  Hydro
  Informer
  Limiters
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  RelativisticEulerSolutions
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )

set(LIBS_TO_LINK_WITH_CONTROL_SYSTEM
  ControlSystem
  ControlSystemMeasurements
  )

function(add_ghmhd_executable SUFFIX INITIAL_DATA USE_CONTROL_SYSTEMS)
  set(EXECUTABLE "EvolveGhValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhValenciaDivClean.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    INITIAL_DATA=${INITIAL_DATA}
    USE_CONTROL_SYSTEMS=${USE_CONTROL_SYSTEMS}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
  if (USE_CONTROL_SYSTEMS STREQUAL "true")
    target_link_libraries(
      ${EXECUTABLE}
      PRIVATE
      ${LIBS_TO_LINK_WITH_CONTROL_SYSTEM}
      )
  endif()
endfunction()

add_ghmhd_executable(
  ""
  evolution::NumericInitialData
  false
  )

add_ghmhd_executable(
  "Bns"
  evolution::NumericInitialData
  true
  )

function(add_ghmhd_with_horizon_executable SUFFIX INITIAL_DATA)
  set(EXECUTABLE "EvolveGhValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhValenciaDivCleanWithHorizon.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    INITIAL_DATA=${INITIAL_DATA}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_ghmhd_with_horizon_executable(
  BondiHoyleAccretion
  gh::Solutions::WrappedGr<grmhd::AnalyticData::BondiHoyleAccretion>
  false
  )

add_ghmhd_with_horizon_executable(
  MagnetizedFmDisk
  gh::Solutions::WrappedGr<grmhd::AnalyticData::MagnetizedFmDisk>
  false
  )

add_ghmhd_with_horizon_executable(
  BondiMichel
  gh::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>
  false
  )

add_ghmhd_with_horizon_executable(
  FishboneMoncriefDisk
  gh::Solutions::WrappedGr<RelativisticEuler::Solutions::FishboneMoncriefDisk>
  false
  )

add_ghmhd_executable(
  CcsnCollapse
  gh::Solutions::WrappedGr<grmhd::AnalyticData::CcsnCollapse>
  false
  )

add_ghmhd_executable(
  TovStar
  gh::Solutions::WrappedGr<RelativisticEuler::Solutions::TovStar>
  false
  )
