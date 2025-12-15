# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarSelfForceEvents")

set(LIBRARY_SOURCES
  Test_ObserveSelfForce.cpp
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
  Interpolation
  LinearOperators
  Observer
  ObserverHelpers
  Parallel
  ScalarSelfForceAnalyticData
  ScalarSelfForceEvents
  Spectral
  Utilities
  )
