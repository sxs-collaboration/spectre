# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE EvolveScalarTensorSingleBlackHole)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveScalarTensorSingleBlackHole.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
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
  ScalarTensor
  ScalarTensorAnalyticData
  ScalarTensorPointwise
  Serialization
  Time
  Utilities
  )
