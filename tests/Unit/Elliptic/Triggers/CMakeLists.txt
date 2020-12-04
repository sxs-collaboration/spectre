# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticTriggers")

set(LIBRARY_SOURCES
  Test_EveryNIterations.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Triggers/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Elliptic;Parallel;EventsAndTriggers;Utilities"
  )
