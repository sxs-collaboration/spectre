# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Time")

set(LIBRARY_SOURCES
  Test_AdaptiveSteppingDiagnostics.cpp
  Test_AdvanceTime.cpp
  Test_ApproximateTime.cpp
  Test_BoundaryHistory.cpp
  Test_ChangeStepSize.cpp
  Test_ChangeTimeStepperOrder.cpp
  Test_ChooseLtsStepSize.cpp
  Test_CleanHistory.cpp
  Test_EvolutionOrdering.cpp
  Test_History.cpp
  Test_LargestStepperError.cpp
  Test_LtsMode.cpp
  Test_RecordTimeStepperData.cpp
  Test_SelfStart.cpp
  Test_Slab.cpp
  Test_SlabRoundingError.cpp
  Test_StepperErrorTolerances.cpp
  Test_Time.cpp
  Test_TimeSequence.cpp
  Test_TimeStepId.cpp
  Test_TimeStepRequest.cpp
  Test_TimeStepRequestProcessor.cpp
  Test_UpdateU.cpp
  )

add_subdirectory(Actions)
add_subdirectory(ChangeSlabSize)
add_subdirectory(OptionTags)
add_subdirectory(StepChoosers)
add_subdirectory(Tags)
add_subdirectory(TimeSteppers)
add_subdirectory(Triggers)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DataStructuresHelpers
  Domain
  DomainStructure
  Evolution
  Imex
  Time
  TimeStepperHelpers
  Utilities
  )
