# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrSelfForceActions")

set(LIBRARY_SOURCES
  Test_InitializeEffectiveSource.cpp
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
  EllipticDg
  Options
  Parallel
  GrSelfForceActions
  GrSelfForceAnalyticData
  Utilities
  )
