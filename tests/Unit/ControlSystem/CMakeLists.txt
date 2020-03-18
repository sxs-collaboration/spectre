# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ControlSystem")

set(LIBRARY_SOURCES
  Test_Averager.cpp
  Test_Controller.cpp
  Test_FuntionOfTimeUpdater.cpp
  Test_TimescaleTuner.cpp
  FoTUpdater_Helper.cpp
  )

add_test_library(
  ${LIBRARY}
  "ControlSystem"
  "${LIBRARY_SOURCES}"
  "Boost::boost;ControlSystem;FunctionsOfTime"
  )
