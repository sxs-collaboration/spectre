# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_define_test_timeout_factor_option(STANDALONE "standalone")

# Helper function to set up a CMake target for a test executable.  It
# can safely be called multiple times for the same executable.
function(add_standalone_test_executable EXECUTABLE_NAME)
  add_dependencies(test-executables ${EXECUTABLE_NAME})

  if (TARGET ${EXECUTABLE_NAME})
    return()
  endif()

  add_spectre_executable(
    ${EXECUTABLE_NAME}
    ${EXECUTABLE_NAME}.cpp
    )

  target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    Catch2::Catch2
    )

  add_dependencies(
    ${EXECUTABLE_NAME}
    module_GlobalCache
    module_Main
    )
endfunction()

# Helper function to set standard test properties for standalone tests.
function(set_standalone_test_properties TEST_NAME)
  spectre_test_timeout(TIMEOUT STANDALONE 10)

  set_tests_properties(
    "${TEST_NAME}"
    PROPERTIES
    TIMEOUT "${TIMEOUT}"
    LABELS "standalone"
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endfunction()

# For tests that result in a failure it is necessary to redirect
# output from stderr to stdout. However, it was necessary at least on
# some systems to do this redirect inside a shell command.
find_program(SHELL_EXECUTABLE "sh")
if (NOT SHELL_EXECUTABLE)
  message(FATAL_ERROR
    "Could not find 'sh' shell to execute standalone failure tests")
endif()

# Add a standalone test named TEST_NAME that runs an executable with
# no arguments.  A test named Foo.Bar.Baz will run the executable
# Test_Baz by default.
#
# A REGEX_TO_MATCH named argument may be passed, in which case the
# test will pass if the output matches it, otherwise the test will
# pass if the executable succeeds without any "ERROR" output.
#
# An EXECUTABLE named argument can be passed to override the
# executable name.
#
# An INPUT_FILE named argument can be passed to pass an input file to
# the executable.
function(add_standalone_test TEST_NAME)
  cmake_parse_arguments(
    ARG
    ""
    "REGEX_TO_MATCH;EXECUTABLE;INPUT_FILE"
    ""
    ${ARGN})

  if(DEFINED ARG_EXECUTABLE)
    set(EXECUTABLE_NAME "${ARG_EXECUTABLE}")
  else()
    # Extract last component of test name as executable name
    string(REGEX MATCH "[^.]*$" EXECUTABLE_NAME "${TEST_NAME}")
    set(EXECUTABLE_NAME "Test_${EXECUTABLE_NAME}")
  endif()

  add_standalone_test_executable("${EXECUTABLE_NAME}")
  if(DEFINED ARG_INPUT_FILE)
    set(INPUT_FILE_ARGS
      "--input-file ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INPUT_FILE}")
  else()
    set(INPUT_FILE_ARGS "")
  endif()
  add_test(
    NAME "${TEST_NAME}"
    COMMAND
    ${SHELL_EXECUTABLE}
    -c
    "${SPECTRE_TEST_RUNNER} ${CMAKE_BINARY_DIR}/bin/${EXECUTABLE_NAME} ${INPUT_FILE_ARGS} 2>&1"
    )

  set_standalone_test_properties("${TEST_NAME}")
  if(NOT DEFINED ARG_REGEX_TO_MATCH)
    set_tests_properties(
      "${TEST_NAME}"
      PROPERTIES
      FAIL_REGULAR_EXPRESSION "ERROR")
  else()
    set_tests_properties(
      "${TEST_NAME}"
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "${ARG_REGEX_TO_MATCH}")
  endif()
endfunction()
