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
    ${LIBS_TO_LINK}
)

add_ghmhd_executable(
  ""
  "evolution::NumericInitialData,false,BondiSachs"
  "${LIBS_TO_LINK}"
  )

add_ghmhd_executable(
  "Bns"
  "evolution::NumericInitialData,true,BondiSachs"
  "${LIBS_TO_LINK_WITH_CONTROL_SYSTEM}"
  )

add_ghmhd_with_horizon_executable(
  BondiHoyleAccretion
  "gh::Solutions::WrappedGr<grmhd::AnalyticData::BondiHoyleAccretion>,false,BondiSachs"
  "${LIBS_TO_LINK};ApparentHorizonFinder;ParallelInterpolation"
  )

add_ghmhd_with_horizon_executable(
  MagnetizedFmDisk
  "gh::Solutions::WrappedGr<grmhd::AnalyticData::MagnetizedFmDisk>,false,BondiSachs"
  "${LIBS_TO_LINK};ApparentHorizonFinder;ParallelInterpolation"
  )

add_ghmhd_with_horizon_executable(
  BondiMichel
  "gh::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>,false,BondiSachs"
  "${LIBS_TO_LINK};ApparentHorizonFinder;ParallelInterpolation"
  )

add_ghmhd_with_horizon_executable(
  FishboneMoncriefDisk
  "gh::Solutions::WrappedGr<RelativisticEuler::Solutions::FishboneMoncriefDisk>,false,BondiSachs"
  "${LIBS_TO_LINK};ApparentHorizonFinder;ParallelInterpolation"
  )

add_ghmhd_executable(
  CcsnCollapse
  "gh::Solutions::WrappedGr<grmhd::AnalyticData::CcsnCollapse>,false,BondiSachs"
  "${LIBS_TO_LINK}"
  )

add_ghmhd_executable(
  TovStar
  "gh::Solutions::WrappedGr<RelativisticEuler::Solutions::TovStar>,false,BondiSachs"
  "${LIBS_TO_LINK}"
  )
