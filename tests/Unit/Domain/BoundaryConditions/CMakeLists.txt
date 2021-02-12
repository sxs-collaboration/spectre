# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainBoundaryConditions")

set(LIBRARY_SOURCES
  Test_BoundaryCondition.cpp
  Test_GenericBcs.cpp
  Test_GetBoundaryConditionsBase.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/BoundaryConditions"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainBoundaryConditions
  DomainBoundaryConditionsHelpers
  Utilities
  )
