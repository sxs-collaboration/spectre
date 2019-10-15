# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Time")

set(LIBRARY_SOURCES
  Test_EvolutionOrdering.cpp
  Test_History.cpp
  Test_Slab.cpp
  Test_Time.cpp
  Test_TimeStepId.cpp
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
  "Time"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
