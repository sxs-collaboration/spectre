# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ElasticityBoundaryConditions")

set(LIBRARY_SOURCES
  Test_Zero.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Elasticity/BoundaryConditions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ElasticityBoundaryConditions
  Elliptic
  Utilities
  )
