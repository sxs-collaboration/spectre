# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionBoundaryConditions")

set(LIBRARY_SOURCES
  Test_Type.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  EvolutionBoundaryConditions
  Utilities
  )
