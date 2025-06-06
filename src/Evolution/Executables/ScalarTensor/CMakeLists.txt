# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Actions
  ApparentHorizonFinder
  Charmxx::main
  ControlSystem
  CoordinateMaps
  CurvedScalarWave
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
  GhScalarTensorAnalyticData
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  ScalarGaussBonnetPointwise
  ScalarTensor
  ScalarTensorAnalyticData
  ScalarTensorPointwise
  Serialization
  Time
  Utilities
  )

set(SINGLE_BH_EXECUTABLE EvolveScalarTensorSingleBlackHole)
add_spectre_executable(
  ${SINGLE_BH_EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveScalarTensorSingleBlackHole.cpp
  )
target_link_libraries(
  ${SINGLE_BH_EXECUTABLE}
  PRIVATE
  ${LIBS_TO_LINK}
  Cce
  )
