# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Actions
  Charmxx::main
  CoordinateMaps
  DiscontinuousGalerkin
  Domain
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativity
  GeneralRelativityAnalyticData
  GeneralRelativitySolutions
  GeneralizedHarmonic
  Importers
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelAmr
  ParallelInterpolation
  PhaseControl
  Printf
  Serialization
  Time
  Utilities
  )

# GH without horizons
function(add_generalized_harmonic_executable_without_horizon DIM)
  set(EXECUTABLE "EvolveGhNoBlackHole${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhNoBlackHole.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_generalized_harmonic_executable_without_horizon(1)
add_generalized_harmonic_executable_without_horizon(2)
add_generalized_harmonic_executable_without_horizon(3)

# Single BH
set(SINGLE_BH_EXECUTABLE EvolveGhSingleBlackHole)
add_spectre_executable(
  ${SINGLE_BH_EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveGhSingleBlackHole.cpp
  )
target_link_libraries(
  ${SINGLE_BH_EXECUTABLE}
  PRIVATE
  ${LIBS_TO_LINK}
  ApparentHorizonFinder
  Cce
  ControlSystem
  Deadlock
  )

# BBH
set(BBH_EXECUTABLE EvolveGhBinaryBlackHole)
add_spectre_executable(
  ${BBH_EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveGhBinaryBlackHole.cpp
)
target_link_libraries(
  ${BBH_EXECUTABLE}
  PRIVATE
  ${LIBS_TO_LINK}
  ApparentHorizonFinder
  Cce
  ControlSystem
  Deadlock
  EvolutionTriggers
  )
if (TARGET SpEC::Exporter)
  target_compile_definitions(
    ${BBH_EXECUTABLE} PRIVATE HAS_SPEC_EXPORTER)
endif()

# GH + characteristic
function(add_gh_and_characteristic_executable EXECUTABLE EVOLVE_CCM)
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveGhAndCharacteristic.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    EVOLVE_CCM=${EVOLVE_CCM}
    )
  target_link_libraries(
    ${EXECUTABLE}
    PRIVATE
    ${LIBS_TO_LINK}
    Cce
    )
endfunction()

add_gh_and_characteristic_executable(EvolveGhCce false)
add_gh_and_characteristic_executable(EvolveGhCcm true)
