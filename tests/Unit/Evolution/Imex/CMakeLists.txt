# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Imex")

set(LIBRARY_SOURCES
  Test_GuessResult.cpp
  Test_Initialize.cpp
  Test_Mode.cpp
  Test_SolveImplicitSector.cpp
  )

add_subdirectory(Actions)
add_subdirectory(Tags)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  Framework
  Imex
  ImexHelpers
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
