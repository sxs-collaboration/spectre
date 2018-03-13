################################################################################
#
# \file      cmake/CodeCoverage.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Setup target for code coverage analysis
# \date      Wed 22 Feb 2017 07:39:09 AM MST
#
# Modifications:
# 1) Change "Quinoa" to "SpECTRE" and "quinoa" to "spectre"
# 2) Split lines to make commands more legible
################################################################################

# ##############################################################################
# Function to add code coverage target
#
# setup_target_for_coverage( <TEST_SUITE> <OUTPUT_PATH> <TARGET_NAME> <TEST_RUNNER>
#                            [TESTRUNNER_ARGS ...]
#                            [DEPENDS dep1 dep2 ... ] )
#
# Mandatory arguments:
# --------------------
#
# TEST_SUITE - Test TEST_SUITE name to be displayed in HTML report title.
#
# OUTPUT_PATH - Path to prepend to where the report is generated:
# <OUTPUT_PATH>${TARGET_NAME}/index.html.
#
# TARGET_NAME - The name of the code coverage target. The HTML report on code
# coverage is generated at the OUTPUT_PATH <OUTPUT_PATH>/${TARGET_NAME}/index.html.
#
# TEST_RUNNER - Command line of the test runner.
#
# Optional arguments:
# -------------------
#
# TESTRUNNER_ARGS arg1 arg2 ... - Optional arguments to test runner. Pass them
# in list form, e.g.: "-v;-g;group" for passing '-v -g group'. Default: "".
#
# DEPENDS dep1 dep2 ... - Optional dependencies added to test coverage target.
# Default: "". Here all dependencies should be given that should be covered by
# the test suite the coverage is being setup for, as well as those that are
# required for successfully building the tests and the test runner.
#
# Author: J. Bakosi
#
# ##############################################################################
function(SETUP_TARGET_FOR_COVERAGE
    TEST_SUITE OUTPUT_PATH TARGET_NAME TEST_RUNNER)
  if (NOT IS_ABSOLUTE ${OUTPUT_PATH})
    set(OUTPUT_PATH ${CMAKE_BINARY_DIR}/${OUTPUT_PATH})
  endif()

  set(multiValueArgs TESTRUNNER_ARGS DEPENDS IGNORE_COV)
  cmake_parse_arguments(
      ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
      ${ARGN})

  if(NOT LCOV)
    MESSAGE(FATAL_ERROR "lcov not found! Aborting...")
  endif()

  if(NOT GENHTML)
    MESSAGE(FATAL_ERROR "genhtml not found! Aborting...")
  endif()

  if(NOT SED)
    MESSAGE(FATAL_ERROR "sed not found! Aborting...")
  endif()

  # Set shortcut for output: OUTPUT_PATH/target
  set(OUTPUT ${OUTPUT_PATH}/${TARGET_NAME})
  file(MAKE_DIRECTORY ${OUTPUT_PATH})

  # Retrieve the git commit hash
  find_package(Git REQUIRED)
  execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_COMMIT_HASH_NO_TAG
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Setup code coverage target
  add_custom_target(
      ${TARGET_NAME}
      # Cleanup any old intermediate data
      COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT}.base.info
      ${OUTPUT}.test.info ${OUTPUT}.total.info
      ${OUTPUT}.filtered.info
      # Cleanup lcov
      COMMAND ${LCOV} --gcov-tool ${GCOV} --directory . --zerocounters
      # Capture initial state yielding zero coverage baseline
      COMMAND ${LCOV} --gcov-tool ${GCOV} --capture --initial
      --directory . --output-file ${OUTPUT}.base.info
      # Run test TEST_SUITE
      COMMAND ${TEST_RUNNER} ${ARG_TESTRUNNER_ARGS}
      # Capture lcov counters
      COMMAND ${LCOV} --gcov-tool ${GCOV} --capture
      --rc lcov_branch_coverage=0 --directory .
      --output-file ${OUTPUT}.test.info
      # Combine trace files
      COMMAND ${LCOV} --gcov-tool ${GCOV} --rc lcov_branch_coverage=0
      --add-tracefile ${OUTPUT}.base.info
      --add-tracefile ${OUTPUT}.test.info
      --output-file ${OUTPUT}.total.info
      # Filter out unwanted files
      COMMAND ${LCOV} --gcov-tool ${GCOV} --rc lcov_branch_coverage=0
      --remove ${OUTPUT}.total.info '*/c++/*' '*/include/*'
      '*/boost/*' '*/charm/*' '*.decl.h' '*.def.h'
      '*/STDIN' '*/tut/*' '*/moduleinit*' '*InfoFromBuild.cpp'
      '${CMAKE_SOURCE_DIR}/src/Executables/*'
      ${ARG_IGNORE_COV}
      --output-file ${OUTPUT}.filtered.info
      # Generate HTML report
      COMMAND ${GENHTML} --legend --demangle-cpp
      --title `cd ${CMAKE_SOURCE_DIR} && git rev-parse HEAD`
      -o ${OUTPUT} ${OUTPUT}.filtered.info
      # Customize page headers in generated html to own
      COMMAND find ${OUTPUT} -type f -print | xargs file | grep text |
      cut -f1 -d: | xargs ${SED} -i'.bak' 's/LCOV - code coverage
      report/SpECTRE ${TEST_SUITE} Test Code Coverage Report/g'
      COMMAND find ${OUTPUT} -type f -name \"*.bak\" -print | xargs file |
      grep text | cut -f1 -d: | xargs rm
      COMMAND find ${OUTPUT} -type f -print | xargs file | grep text |
      cut -f1 -d: | xargs ${SED} -i'.bak'
      's^<td class="headerItem">Test:</td>^<td class="headerItem">Commit:</td>^g'
      COMMAND find ${OUTPUT} -type f -name \"*.bak\" -print | xargs file |
      grep text | cut -f1 -d: | xargs rm
      COMMAND find ${OUTPUT} -type f -print | xargs file | grep text |
      cut -f1 -d:
      | xargs ${SED} -i'.bak' 's^<td class="headerValue">\\\([a-z0-9]\\{40\\}\\\)^
      <td class="headerValue"><a target="_blank"
      href="https://github.com/sxs-collaboration/spectre/commit/\\1">\\1</a>^g'
      # Delete backup files created by sed
      COMMAND find ${OUTPUT} -type f -name \"*.bak\" -print | xargs file |
      grep text | cut -f1 -d: | xargs rm
      # Cleanup any intermediate data
      COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT}.base.info
      ${OUTPUT}.test.info ${OUTPUT}.total.info
      # Copy output into coverage.info to be used by codecov
      COMMAND mv ${OUTPUT}.filtered.info ${CMAKE_BINARY_DIR}/tmp/coverage.info
      # Set work directory for target
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      # Echo what is being done
      COMMENT "SpECTRE ${TEST_SUITE} Test Code Coverage Report"
  )

  # Make test coverage target dependent on optional dependencies passed in using
  # keyword DEPENDS
  add_dependencies(${TARGET_NAME} ${ARG_DEPENDS})

  # Output code coverage target enabled
  string(REPLACE ";" " " ARGUMENTS "${ARG_TESTRUNNER_ARGS}")
  message(
      STATUS
      "Enabling code coverage target '${TARGET_NAME}' tested by "
      "'${TEST_RUNNER} ${ARGUMENTS}', dependencies {${ARG_DEPENDS}}, "
      "report at ${OUTPUT}/index.html"
  )

endfunction()
