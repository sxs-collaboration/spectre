# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_ghmhd_executable INITIAL_DATA_NAME INITIAL_DATA LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGhValenciaDivClean${INITIAL_DATA_NAME}"
    EvolveGhValenciaDivClean
    Evolution/Executables/GrMhd/GhValenciaDivClean
    "EvolutionMetavars<${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_ghmhd_executable)

function(add_ghmhd_with_horizon_executable INITIAL_DATA_NAME INITIAL_DATA LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGhValenciaDivClean${INITIAL_DATA_NAME}"
    EvolveGhValenciaDivCleanWithHorizon
    Evolution/Executables/GrMhd/GhValenciaDivClean
    "EvolutionMetavars<${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_ghmhd_with_horizon_executable)

set(LIBS_TO_LINK
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
  PhaseControl
  RelativisticEulerSolutions
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )

add_ghmhd_executable(
  ""
  evolution::NumericInitialData
  "${LIBS_TO_LINK}"
  )

if (TARGET SpEC::Exporter)
  target_compile_definitions(
    EvolveGhValenciaDivClean PRIVATE HAS_SPEC_EXPORTER)
endif()

add_ghmhd_with_horizon_executable(
  BondiHoyleAccretion
  gh::Solutions::WrappedGr<grmhd::AnalyticData::BondiHoyleAccretion>
  "${LIBS_TO_LINK}"
  )

add_ghmhd_with_horizon_executable(
  MagnetizedFmDisk
  gh::Solutions::WrappedGr<grmhd::AnalyticData::MagnetizedFmDisk>
  "${LIBS_TO_LINK}"
  )

add_ghmhd_with_horizon_executable(
  BondiMichel
  gh::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>
  "${LIBS_TO_LINK};ParallelInterpolation"
  )

add_ghmhd_with_horizon_executable(
  FishboneMoncriefDisk
  gh::Solutions::WrappedGr<RelativisticEuler::Solutions::FishboneMoncriefDisk>
  "${LIBS_TO_LINK}"
  )

add_ghmhd_executable(
  TovStar
  gh::Solutions::WrappedGr<RelativisticEuler::Solutions::TovStar>
  "${LIBS_TO_LINK}"
  )
