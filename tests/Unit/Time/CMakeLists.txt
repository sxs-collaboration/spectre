# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Time")

set(LIBRARY_SOURCES
  Test_EvolutionOrdering.cpp
  Test_History.cpp
  Test_Slab.cpp
  Test_Tags.cpp
  Test_Time.cpp
  Test_TimeStepId.cpp
  Test_Utilities.cpp
  )

add_subdirectory(Actions)
add_subdirectory(StepChoosers)
add_subdirectory(StepControllers)
add_subdirectory(TimeSteppers)
add_subdirectory(Triggers)

add_test_library(
  ${LIBRARY}
  "Time"
  "${LIBRARY_SOURCES}"
  "Boost::boost;Time;TimeStepperHelpers"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
