# Distributed under the MIT License.
# See LICENSE.txt for details.

string(
    REGEX REPLACE
    "^${CMAKE_SOURCE_DIR}/([^/]*)/?.*$" "\\1"
    SUBDIR_OF_SOURCE_DIR ${CMAKE_BINARY_DIR}
)
set(PROHIBITED_SUBDIRS "cmake" "docs" "src" "tests" "tools")

if ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}" OR
    ${SUBDIR_OF_SOURCE_DIR} IN_LIST PROHIBITED_SUBDIRS)
  message(FATAL_ERROR "\n"
      "You attempted to build ${PROJECT_NAME} in the directory:\n"
      "  ${CMAKE_BINARY_DIR}\n"
      "In-source builds, however, are not allowed. "
      "Please create a directory and run cmake from there, passing the path "
      "to the source directory as the last argument; for example:\n"
      "  cd ${CMAKE_SOURCE_DIR}\n"
      "  mkdir build\n"
      "  cd build\n"
      "  cmake [OPTIONS] ${CMAKE_SOURCE_DIR}\n"
      "You also need to remove the CMakeCache.txt file and the "
      "CMakeFiles directory in the source directory, or you will trigger "
      "this error again, even when doing an out-of-source build. Run:\n"
      "  rm -r ${CMAKE_BINARY_DIR}/CMakeCache.txt "
      "${CMAKE_BINARY_DIR}/CMakeFiles\n"
  )
endif()
