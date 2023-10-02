# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  ApparentHorizonFinder
  Charmxx::main
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
  Interpolation
  Limiters
  LinearOperators
  MathFunctions
  ParallelInterpolation
  RelativisticEulerSolutions
  Observer
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )

function(add_grmhd_executable SUFFIX INITIAL_DATA INTERPOLATION_TARGET)
  set(EXECUTABLE "EvolveValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveValenciaDivClean.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    INITIAL_DATA=${INITIAL_DATA}
    INTERPOLATION_TARGET=${INTERPOLATION_TARGET}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_grmhd_executable(
  FishboneMoncriefDisk
  RelativisticEuler::Solutions::FishboneMoncriefDisk
  KerrHorizon
  )

add_grmhd_executable(
  TovStar
  RelativisticEuler::Solutions::TovStar
  CenterOfStar
  )

add_grmhd_executable(
  AlfvenWave
  grmhd::Solutions::AlfvenWave
  ""
  )

add_grmhd_executable(
  BondiMichel
  grmhd::Solutions::BondiMichel
  ""
  )

add_grmhd_executable(
  KomissarovShock
  grmhd::Solutions::KomissarovShock
  ""
  )

add_grmhd_executable(
  SmoothFlow
  grmhd::Solutions::SmoothFlow
  ""
  )

add_grmhd_executable(
  BondiHoyleAccretion
  grmhd::AnalyticData::BondiHoyleAccretion
  ""
  )

add_grmhd_executable(
  BlastWave
  grmhd::AnalyticData::BlastWave
  ""
  )

add_grmhd_executable(
  CcsnCollapse
  grmhd::AnalyticData::CcsnCollapse
  ""
  )

add_grmhd_executable(
  KhInstability
  grmhd::AnalyticData::KhInstability
  ""
  )

add_grmhd_executable(
  MagneticFieldLoop
  grmhd::AnalyticData::MagneticFieldLoop
  ""
  )

add_grmhd_executable(
  MagneticRotor
  grmhd::AnalyticData::MagneticRotor
  ""
  )

add_grmhd_executable(
  MagnetizedFmDisk
  grmhd::AnalyticData::MagnetizedFmDisk
  ""
  )

add_grmhd_executable(
  MagnetizedTovStar
  grmhd::AnalyticData::MagnetizedTovStar
  CenterOfStar
  )

add_grmhd_executable(
  OrszagTangVortex
  grmhd::AnalyticData::OrszagTangVortex
  ""
  )

add_grmhd_executable(
  PolarMagnetizedFmDisk
  grmhd::AnalyticData::PolarMagnetizedFmDisk
  ""
  )

add_grmhd_executable(
  RiemannProblem
  grmhd::AnalyticData::RiemannProblem
  ""
  )

add_grmhd_executable(
  RotatingStar
  RelativisticEuler::Solutions::RotatingStar
  ""
  )

add_grmhd_executable(
  SlabJet
  grmhd::AnalyticData::SlabJet
  ""
  )
