# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionBoundaryConditions")

set(LIBRARY_SOURCES
  Test_Type.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/BoundaryConditions"
  "${LIBRARY_SOURCES}"
  "EvolutionBoundaryConditions"
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  EvolutionBoundaryConditions
  Utilities
  )
