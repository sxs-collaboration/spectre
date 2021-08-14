# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticBoundaryConditions")

set(LIBRARY_SOURCES
  Tags/Test_BoundaryFields.cpp
  Test_AnalyticSolution.cpp
  Test_BoundaryCondition.cpp
  Test_BoundaryConditionType.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/BoundaryConditions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainBoundaryConditions
  DomainStructure
  Elliptic
  Options
  Parallel
  Spectral
  Utilities
  )
