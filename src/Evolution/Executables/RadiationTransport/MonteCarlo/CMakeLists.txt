# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE EvolveMonteCarlo)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveMonteCarlo.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Actions
  Charmxx::main
  CoordinateMaps
  DataStructures
  DgSubcell
  DiscontinuousGalerkin
  DomainCreators
  Events
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativity
  GeneralRelativitySolutions
  GrMhdAnalyticData
  GrMhdSolutions
  H5
  Hydro
  Informer
  LinearOperators
  MathFunctions
  MonteCarlo
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  ValenciaDivClean
  )

