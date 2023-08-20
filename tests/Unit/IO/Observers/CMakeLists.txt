# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Observer")

set(LIBRARY_SOURCES
  Test_ArrayComponentId.cpp
  Test_GetLockPointer.cpp
  Test_Initialize.cpp
  Test_ObservationId.cpp
  Test_ReductionObserver.cpp
  Test_RegisterElements.cpp
  Test_RegisterEvents.cpp
  Test_RegisterSingleton.cpp
  Test_Tags.cpp
  Test_TypeOfObservation.cpp
  Test_VolumeObserver.cpp
  Test_WriteSimpleData.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  DataStructuresHelpers
  DomainStructure
  EventsAndTriggers
  ErrorHandling
  IoTestHelpers
  Observer
  ObserverHelpers
  Options
  Parallel
  Utilities
  Spectral
  )
