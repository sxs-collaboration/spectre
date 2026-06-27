# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelInterpolation")

set(LIBRARY_SOURCES
  Test_ComputeExcisionBoundaryVolumeQuantities.cpp
  Test_ElementReceiveInterpPoints.cpp
  Test_InitializeInterpolationTarget.cpp
  Test_InterpolateWithoutInterpComponent.cpp
  Test_InterpolationTargetKerrHorizon.cpp
  Test_InterpolationTargetLineSegment.cpp
  Test_InterpolationTargetVarsFromElement.cpp
  Test_InterpolationTargetSpecifiedPoints.cpp
  Test_InterpolationTargetSphere.cpp
  Test_InterpolationTargetWedgeSectionTorus.cpp
  Test_ObserveLineSegment.cpp
  Test_ObserveTimeSeriesAndSurfaceData.cpp
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
  GrSurfaces
  H5
  Interpolation
  InterpolationHelpers
  IoTestHelpers
  Logging
  MathFunctions
  Observer
  ParallelInterpolation
  RelativisticEulerSolutions
  Spectral
  SphericalHarmonics
  Strahlkorper
  StrahlkorperIO
  Utilities
  )
