# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EquationsOfState")

set(LIBRARY_SOURCES
  Test_Polytrope.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/EquationsOfState/"
  "${LIBRARY_SOURCES}"
  "EquationsOfState"
  )
