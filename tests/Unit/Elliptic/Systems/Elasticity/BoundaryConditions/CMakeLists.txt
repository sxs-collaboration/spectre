# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ElasticityBoundaryConditions")

set(LIBRARY_SOURCES
  Test_LaserBeam.cpp
  Test_Zero.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainStructure
  ElasticityBoundaryConditions
  ElasticitySolutions
  Elliptic
  Utilities
  )
