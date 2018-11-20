# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Time")

set(LIBRARY_SOURCES
  Test_EveryNSlabs.cpp
  Test_EvolutionOrdering.cpp
  Test_History.cpp
  Test_PastTime.cpp
  Test_Slab.cpp
  Test_SpecifiedSlabs.cpp
  Test_Time.cpp
  Test_TimeId.cpp
  )

add_subdirectory(Actions)
add_subdirectory(StepChoosers)
add_subdirectory(StepControllers)
add_subdirectory(TimeSteppers)

add_test_library(
  ${LIBRARY}
  "Time"
  "${LIBRARY_SOURCES}"
  "Time"
  )
