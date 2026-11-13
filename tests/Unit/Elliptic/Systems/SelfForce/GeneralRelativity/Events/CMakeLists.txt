# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrSelfForceEvents")

set(LIBRARY_SOURCES
  Test_ObserveFlux.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  Elliptic
  Framework
  GrSelfForce
  GrSelfForceAnalyticData
  GrSelfForceEvents
  Observer
  ObserverHelpers
  Parallel
  Spectral
  Utilities
  )
