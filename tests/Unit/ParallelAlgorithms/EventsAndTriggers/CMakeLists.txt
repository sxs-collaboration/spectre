# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EventsAndTriggers")

set(LIBRARY_SOURCES
  Test_Completion.cpp
  Test_EventsAndTriggers.cpp
  Test_RunEventsOnFailure.cpp
  Test_Tags.cpp
  Test_WhenToCheck.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  EventsAndTriggers
  ErrorHandling
  Parallel
  Time
  Utilities
  )
