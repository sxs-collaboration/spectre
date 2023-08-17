# Distributed under the MIT License.
# See LICENSE.txt for details.

# Add an executable with Catch2 unit tests and register them with CTest.
#
# This function is named `add_test_library` although it adds an executable for
# historical reasons. For the same reasons it takes two unused arguments. We can
# refactor this in the future.
#
# Parameters:
# - TEST_EXEC_NAME The name of the test executable to add
# - SOURCE_FILES The source files to compile into the executable
function(add_test_library TEST_EXEC_NAME UNUSED SOURCE_FILES UNUSED2)
  cmake_parse_arguments(ARG WITH_CHARM "" "" ${ARGN})

  add_spectre_executable(
    ${TEST_EXEC_NAME}
    ${SOURCE_FILES}
    )
  add_dependencies(unit-tests ${TEST_EXEC_NAME})
  target_link_libraries(
    ${TEST_EXEC_NAME}
    PRIVATE
    Catch2::Catch2
    Framework
    )

  # Use either the Charm++ or the non-Charm++ TestMain
  if (ARG_WITH_CHARM)
    target_sources(
      ${TEST_EXEC_NAME}
      PRIVATE
      ${CMAKE_SOURCE_DIR}/tests/Unit/TestMainCharm.cpp
    )
    add_dependencies(
      ${TEST_EXEC_NAME}
      module_TestMainCharm
      )
  else()
    target_sources(
      ${TEST_EXEC_NAME}
      PRIVATE
      ${CMAKE_SOURCE_DIR}/tests/Unit/TestMain.cpp
    )
  endif()

  # Register the tests with CTest.
  # We may want to switch to `catch_discover_tests` and remove our own test
  # parsing code in the future. Before we do, we should check if it supports all
  # the features we use.
  # catch_discover_tests(${TEST_EXEC_NAME})
  spectre_add_catch_tests(${TEST_EXEC_NAME})
endfunction()

# Shared test libs are currently unsupported on macOS because Catch2 static
# variables are not properly initialized. This is only needed for test helper
# libraries with Catch2 assertions in them.
set(SPECTRE_TEST_LIBS_TYPE "")
if(APPLE)
  set(SPECTRE_TEST_LIBS_TYPE STATIC)
endif()
