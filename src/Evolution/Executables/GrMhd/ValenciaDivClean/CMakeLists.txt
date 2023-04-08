# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_grmhd_executable INITIAL_DATA_NAME INITIAL_DATA LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveValenciaDivClean${INITIAL_DATA_NAME}"
    EvolveValenciaDivClean
    Evolution/Executables/GrMhd/ValenciaDivClean
    "EvolutionMetavars<${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_grmhd_executable)

set(LIBS_TO_LINK
  CoordinateMaps
  DgSubcell
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FiniteDifference
  GeneralRelativitySolutions
  GrMhdAnalyticData
  GrMhdSolutions
  Hydro
  Informer
  Limiters
  LinearOperators
  MathFunctions
  ParallelInterpolation
  RelativisticEulerSolutions
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )

add_grmhd_executable(
  FishboneMoncriefDisk
  RelativisticEuler::Solutions::FishboneMoncriefDisk,KerrHorizon
  "${LIBS_TO_LINK};ApparentHorizons;Interpolation;ParallelInterpolation"
  )

add_grmhd_executable(
  TovStar
  RelativisticEuler::Solutions::TovStar,CenterOfStar
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  AlfvenWave
  grmhd::Solutions::AlfvenWave
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  BondiMichel
  grmhd::Solutions::BondiMichel
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  KomissarovShock
  grmhd::Solutions::KomissarovShock
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  SmoothFlow
  grmhd::Solutions::SmoothFlow
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  BondiHoyleAccretion
  grmhd::AnalyticData::BondiHoyleAccretion
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  BlastWave
  grmhd::AnalyticData::BlastWave
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  CcsnCollapse
  grmhd::AnalyticData::CcsnCollapse
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  KhInstability
  grmhd::AnalyticData::KhInstability
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  MagneticFieldLoop
  grmhd::AnalyticData::MagneticFieldLoop
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  MagneticRotor
  grmhd::AnalyticData::MagneticRotor
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  MagnetizedFmDisk
  grmhd::AnalyticData::MagnetizedFmDisk
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  MagnetizedTovStar
  grmhd::AnalyticData::MagnetizedTovStar,CenterOfStar
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  OrszagTangVortex
  grmhd::AnalyticData::OrszagTangVortex
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  RiemannProblem
  grmhd::AnalyticData::RiemannProblem
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  RotatingStar
  RelativisticEuler::Solutions::RotatingStar
  "${LIBS_TO_LINK}"
  )

add_grmhd_executable(
  SlabJet
  grmhd::AnalyticData::SlabJet
  "${LIBS_TO_LINK}"
  )
