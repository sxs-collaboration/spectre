# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_ghmhd_executable NAME METAVARS LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGhValenciaDivClean${NAME}"
    EvolveGhValenciaDivClean
    Evolution/Executables/GrMhd/GhValenciaDivClean
    "EvolutionMetavars<${METAVARS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_ghmhd_executable)

function(add_ghmhd_with_horizon_executable NAME METAVARS LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "EvolveGhValenciaDivCleanWithHorizon${NAME}"
    EvolveGhValenciaDivCleanWithHorizon
    Evolution/Executables/GrMhd/GhValenciaDivClean
    "EvolutionMetavars<${METAVARS}>"
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
  "false,BondiSachs"
  "${LIBS_TO_LINK}"
  )

add_ghmhd_executable(
  "Bns"
  "true,BondiSachs"
  "${LIBS_TO_LINK_WITH_CONTROL_SYSTEM}"
  )

if (TARGET SpEC::Exporter)
  target_compile_definitions(
    EvolveGhValenciaDivClean PRIVATE HAS_SPEC_EXPORTER)
  target_compile_definitions(
    EvolveGhValenciaDivCleanBns PRIVATE HAS_SPEC_EXPORTER)
endif()

add_ghmhd_with_horizon_executable(
  ""
  "false,BondiSachs"
  "${LIBS_TO_LINK};ApparentHorizons;ParallelInterpolation"
  )
