# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelInterpolation")

set(LIBRARY_SOURCES
  Test_AddTemporalIdsToInterpolationTarget.cpp
  Test_AngularOrdering.cpp
  Test_CleanUpInterpolator.cpp
  Test_ComputeDestVars.cpp
  Test_ElementReceiveInterpPoints.cpp
  Test_InitializeInterpolationTarget.cpp
  Test_InitializeInterpolator.cpp
  Test_Interpolate.cpp
  Test_InterpolateWithoutInterpComponent.cpp
  Test_InterpolationTargetKerrHorizon.cpp
  Test_InterpolationTargetLineSegment.cpp
  Test_InterpolationTargetReceiveVars.cpp
  Test_InterpolationTargetVarsFromElement.cpp
  Test_InterpolationTargetSpecifiedPoints.cpp
  Test_InterpolationTargetSphere.cpp
  Test_InterpolationTargetWedgeSectionTorus.cpp
  Test_InterpolatorReceiveAndDumpVolumeData.cpp
  Test_InterpolatorReceivePoints.cpp
  Test_InterpolatorRegisterElement.cpp
  Test_ObserveLineSegment.cpp
  Test_ObserveTimeSeriesAndSurfaceData.cpp
  Test_ParallelInterpolator.cpp
  Test_Protocols.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  ControlSystem
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  Events
  GeneralRelativitySolutions
  H5
  Interpolation
  IoTestHelpers
  Logging
  MathFunctions
  Observer
  ParallelInterpolation
  RelativisticEulerSolutions
  Spectral
  SphericalHarmonics
  StrahlkorperGr
  Utilities
  )
