# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrSelfForceAmrCriteria")

set(LIBRARY_SOURCES
  Test_RefineAtBoundary.cpp
  Test_RefineAtPunctures.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  AmrCriteria
  DataStructures
  DomainCreators
  DomainStructure
  Options
  Parallel
  GrSelfForceAmrCriteria
  GrSelfForceAnalyticData
  Utilities
  )
