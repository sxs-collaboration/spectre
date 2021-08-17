# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelAlgorithmsEvents")

set(LIBRARY_SOURCES
  Test_ObserveErrorNorms.cpp
  Test_ObserveFields.cpp
  Test_ObserveNorms.cpp
  Test_ObserveTimeStep.cpp
  Test_ObserveVolumeIntegrals.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Events/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  ErrorHandling
  Events
  EventsAndTriggers
  Interpolation
  IO
  Spectral
  Time
  Utilities
  )
