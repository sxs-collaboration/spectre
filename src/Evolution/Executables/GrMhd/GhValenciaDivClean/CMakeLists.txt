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

function(add_ghmhd_executable SUFFIX USE_CONTROL_SYSTEMS)
  set(EXECUTABLE "EvolveGhValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhValenciaDivClean.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
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
  "false"
  )

add_ghmhd_executable(
  "Bns"
   "true"
  )

function(add_ghmhd_with_horizon_executable SUFFIX)
  set(EXECUTABLE "EvolveGhValenciaDivClean${SUFFIX}")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhValenciaDivCleanWithHorizon.cpp
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_ghmhd_with_horizon_executable(
  "WithHorizon"
  "false"
  )
