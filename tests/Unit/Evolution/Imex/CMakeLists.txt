# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Imex")

set(LIBRARY_SOURCES
  Test_GuessResult.cpp
  Test_Mode.cpp
  )

add_subdirectory(Tags)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  Framework
  Imex
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
