# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ApparentHorizonFinder")

set(LIBRARY_SOURCES
  Test_ApparentHorizonFinder.cpp
  Test_CleanUp.cpp
  Test_ComputeCoords.cpp
  Test_ComputeExcisionBoundaryVolumeQuantities.cpp
  Test_ComputeHorizonVolumeQuantities.cpp
  Test_ComputeVarsToInterpolateToTarget.cpp
  Test_CurrentTime.cpp
  Test_Destination.cpp
  Test_FastFlow.cpp
  Test_FindApparentHorizon.cpp
  Test_Initialization.cpp
  Test_InterpolateVolumeVars.cpp
  Test_InterpolationTargetApparentHorizon.cpp
  Test_ObserveCenters.cpp
  Test_OptionTags.cpp
  Test_Storage.cpp
  Test_Tags.cpp
  )

add_subdirectory(Callbacks)
add_subdirectory(Events)
add_subdirectory(Protocols)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ApparentHorizonFinder
  ApparentHorizonFinderCriteria
  FiniteDifference
  GeneralRelativity
  GeneralRelativitySolutions
  GrSurfacesHelpers
  InterpolationHelpers
  IoTestHelpers
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

add_subdirectory(Criteria)
add_subdirectory(Python)
