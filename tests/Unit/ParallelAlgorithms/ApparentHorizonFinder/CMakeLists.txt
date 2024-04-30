# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ApparentHorizonFinder")

set(LIBRARY_SOURCES
  Test_ApparentHorizonFinder.cpp
  Test_ComputeExcisionBoundaryVolumeQuantities.cpp
  Test_ComputeHorizonVolumeQuantities.cpp
  Test_FastFlow.cpp
  Test_InterpolationTargetApparentHorizon.cpp
  Test_ObserveCenters.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ApparentHorizonFinder
  FiniteDifference
  GeneralRelativitySolutions
  InterpolationHelpers
  LinearOperators
  Logging
  ObserverHelpers
  Options
  ParallelInterpolation
  RootFinding
  SphericalHarmonics
  SphericalHarmonicsHelpers
  Time
  Utilities
  )

add_subdirectory(Python)
