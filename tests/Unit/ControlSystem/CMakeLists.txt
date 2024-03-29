# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ControlSystem")

set(LIBRARY_SOURCES
  Test_Averager.cpp
  Test_CombinedName.cpp
  Test_Controller.cpp
  Test_DataVectorHelpers.cpp
  Test_EventTriggerMetafunctions.cpp
  Test_ExpirationTimes.cpp
  Test_FutureMeasurements.cpp
  Test_IsSize.cpp
  Test_Measurements.cpp
  Test_Metafunctions.cpp
  Test_RunCallbacks.cpp
  Test_Tags.cpp
  Test_TimescaleTuner.cpp
  Test_Trigger.cpp
  Test_UpdateFunctionOfTime.cpp
  Test_WriteData.cpp
  )

add_subdirectory(Actions)
add_subdirectory(ControlErrors)
add_subdirectory(Protocols)
add_subdirectory(Systems)
add_subdirectory(Tags)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ApparentHorizonFinder
  ControlSystem
  ControlSystemMeasurements
  EventsAndDenseTriggers
  FunctionsOfTime
  H5
  Observer
  ObserverHelpers
  Parallel
  PostNewtonianHelpers
  SphericalHarmonics
  Time
  Utilities
)
