# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ControlSystem")

set(LIBRARY_SOURCES
  Test_Averager.cpp
  Test_Controller.cpp
  Test_EventTriggerMetafunctions.cpp
  Test_InitialExpirationTimes.cpp
  Test_Measurement.cpp
  Test_Metafunctions.cpp
  Test_Tags.cpp
  Test_TimescaleTuner.cpp
  Test_Trigger.cpp
  Test_UpdateFunctionOfTime.cpp
  )

add_subdirectory(Actions)
add_subdirectory(Protocols)

add_test_library(
  ${LIBRARY}
  "ControlSystem"
  "${LIBRARY_SOURCES}"
  "Boost::boost;ControlSystem;Domain;FunctionsOfTime"
  )
