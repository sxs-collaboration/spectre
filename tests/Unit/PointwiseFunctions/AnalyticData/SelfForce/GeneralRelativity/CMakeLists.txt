# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrSelfForceAnalyticData")

set(LIBRARY_SOURCES
  Test_CircularOrbit.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Utilities
  GrSelfForce
  GrSelfForceAnalyticData
  Domain
  DomainCreators
  LinearOperators
  )
