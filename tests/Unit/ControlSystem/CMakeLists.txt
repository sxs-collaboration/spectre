# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ControlSystem")

set(LIBRARY_SOURCES
  Test_Controller.cpp
  Test_PiecewisePolynomial.cpp
  Test_SettleToConstant.cpp
  Test_TimescaleTuner.cpp
  )

add_test_library(
  ${LIBRARY}
  "ControlSystem"
  "${LIBRARY_SOURCES}"
  "ControlSystem"
  )
