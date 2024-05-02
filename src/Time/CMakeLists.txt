# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Time)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AdaptiveSteppingDiagnostics.cpp
  ApproximateTime.cpp
  ChooseLtsStepSize.cpp
  History.cpp
  LargestStepperError.cpp
  SelfStart.cpp
  Slab.cpp
  StepperErrorEstimate.cpp
  StepperErrorTolerances.cpp
  Time.cpp
  TimeSequence.cpp
  TimeStepId.cpp
  TimeStepRequest.cpp
  Utilities.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdaptiveSteppingDiagnostics.hpp
  ApproximateTime.hpp
  BoundaryHistory.hpp
  ChooseLtsStepSize.hpp
  EvolutionOrdering.hpp
  History.hpp
  LargestStepperError.hpp
  SelfStart.hpp
  Slab.hpp
  StepperErrorEstimate.hpp
  StepperErrorTolerances.hpp
  TakeStep.hpp
  Time.hpp
  TimeSequence.hpp
  TimeStepId.hpp
  TimeStepRequest.hpp
  Utilities.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  ErrorHandling
  EventsAndTriggers
  Options
  Serialization
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(ChangeSlabSize)
add_subdirectory(OptionTags)
add_subdirectory(StepChoosers)
add_subdirectory(Tags)
add_subdirectory(TimeSteppers)
add_subdirectory(Triggers)
