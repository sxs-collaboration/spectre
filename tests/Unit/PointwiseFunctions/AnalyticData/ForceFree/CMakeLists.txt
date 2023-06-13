# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ForceFreeAnalyticData")

set(LIBRARY_SOURCES
  Test_FfeBreakdown.cpp
  Test_RotatingDipole.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ForceFreeAnalyticData
  Options
  Utilities
)
