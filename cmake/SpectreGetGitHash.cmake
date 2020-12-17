# Distributed under the MIT License.
# See LICENSE.txt for details.

set(GIT_HASH "")
set(GIT_BRANCH "")
set(GIT_DESCRIPTION "")

if(EXISTS ${CMAKE_SOURCE_DIR}/.git)
  find_package(Git)

  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_DESCRIPTION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    message(STATUS "Git description: ${GIT_DESCRIPTION}")
  endif()
else()
  message(STATUS "Not running in a git repository. Some features are disabled.")
endif()
