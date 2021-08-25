# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticTriggers")

set(LIBRARY_SOURCES
  Test_EveryNIterations.cpp
  Test_HasConverged.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Triggers/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Convergence
  DataStructures
  Elliptic
  EventsAndTriggers
  Parallel
  Utilities
  )
