# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PostNewtonianHelpers")

set(LIBRARY_SOURCES
  Test_BinaryTrajectories.cpp
  )

add_test_library(
  ${LIBRARY}
  "Helpers/Tests/PointwiseFunctions/PostNewtonian/"
  "${LIBRARY_SOURCES}"
  "PostNewtonianHelpers"
  )
