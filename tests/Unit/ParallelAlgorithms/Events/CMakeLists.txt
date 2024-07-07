# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelAlgorithmsEvents")

set(LIBRARY_SOURCES
  Test_ErrorIfDataTooBig.cpp
  Test_ObserveAdaptiveSteppingDiagnostics.cpp
  Test_ObserveAtExtremum.cpp
  Test_ObserveFields.cpp
  Test_ObserveNorms.cpp
  Test_ObserveTimeStep.cpp
  Test_ObserveTimeStepVolume.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  ErrorHandling
  Events
  EventsAndTriggers
  EventsHelpers
  H5
  Interpolation
  Observer
  Spectral
  Time
  Utilities
  )
