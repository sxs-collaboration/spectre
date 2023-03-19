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
  RelativisticEulerSolutions
  Time
  Utilities
  ValenciaDivClean
  )

add_ghmhd_executable(
  ""
  evolution::NumericInitialData
  "${LIBS_TO_LINK}"
  )

add_ghmhd_with_horizon_executable(
  BondiHoyleAccretion
  GeneralizedHarmonic::Solutions::WrappedGr<grmhd::AnalyticData::BondiHoyleAccretion>
  "${LIBS_TO_LINK}"
  )

add_ghmhd_with_horizon_executable(
  MagnetizedFmDisk
  GeneralizedHarmonic::Solutions::WrappedGr<grmhd::AnalyticData::MagnetizedFmDisk>
  "${LIBS_TO_LINK}"
  )

add_ghmhd_with_horizon_executable(
  BondiMichel
  GeneralizedHarmonic::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>
  "${LIBS_TO_LINK};ParallelInterpolation"
  )

add_ghmhd_with_horizon_executable(
  FishboneMoncriefDisk
  GeneralizedHarmonic::Solutions::WrappedGr<RelativisticEuler::Solutions::FishboneMoncriefDisk>
  "${LIBS_TO_LINK}"
  )

add_ghmhd_executable(
  TovStar
  GeneralizedHarmonic::Solutions::WrappedGr<RelativisticEuler::Solutions::TovStar>
  "${LIBS_TO_LINK}"
  )
