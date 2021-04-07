# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_INPUT_FILE_TEST_TIMEOUT_FACTOR
  "Multiply timeout for input file tests by this factor"
  1)

find_package(Python REQUIRED)

function(add_single_input_file_test INPUT_FILE EXECUTABLE COMMAND_LINE_ARGS
                                    CHECK_TYPE TIMEOUT)
  # Extract just the name of the input file
  get_filename_component(INPUT_FILE_NAME "${INPUT_FILE}" NAME)

  # Extract the main subdirectory name
  string(FIND "${INPUT_FILE}" "tests/InputFiles/" POSITION_OF_INPUT_FILE_DIR)
  math(EXPR
    POSITION_OF_INPUT_FILE_DIR
    ${POSITION_OF_INPUT_FILE_DIR}+17
    # 17 is the length of "tests/InputFiles/"
    )
  string(SUBSTRING "${INPUT_FILE}" ${POSITION_OF_INPUT_FILE_DIR}
    -1 TEMP)
  string(FIND "${TEMP}" "/" POSITION_OF_SLASH)
  string(SUBSTRING "${TEMP}" 0 ${POSITION_OF_SLASH}
      EXECUTABLE_DIR_NAME)

  # Set tags for the test
  set(TAGS "InputFiles;${EXECUTABLE_DIR_NAME};${CHECK_TYPE}")
  string(TOLOWER "${TAGS}" TAGS)

  set(
    CTEST_NAME
    "\"InputFiles.${EXECUTABLE_DIR_NAME}.${INPUT_FILE_NAME}.${CHECK_TYPE}\""
    )
  set(
    RUN_DIRECTORY
    "${EXECUTABLE_DIR_NAME}.${INPUT_FILE_NAME}.${CHECK_TYPE}"
    )
  if ("${CHECK_TYPE}" STREQUAL "parse")
    add_test(
      NAME "${CTEST_NAME}"
      COMMAND ${CMAKE_BINARY_DIR}/bin/${EXECUTABLE}
      --check-options --input-file ${INPUT_FILE}
      )
  elseif("${CHECK_TYPE}" STREQUAL "execute")
    add_test(
      NAME "${CTEST_NAME}"
      # This script is written below, and only once
      COMMAND sh ${PROJECT_BINARY_DIR}/tmp/InputFileExecuteAndClean.sh
      ${EXECUTABLE} ${INPUT_FILE} ${COMMAND_LINE_ARGS}
      # Make sure we run the test in the build directory for cleaning its output
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
  elseif("${CHECK_TYPE}" STREQUAL "execute_check_output")
    add_test(
      NAME "${CTEST_NAME}"
      # This script is written below, and only once
      COMMAND sh ${PROJECT_BINARY_DIR}/tmp/ExecuteCheckOutputFilesAndClean.sh
      ${EXECUTABLE} ${INPUT_FILE} ${RUN_DIRECTORY} "${COMMAND_LINE_ARGS}"
      # Make sure we run the test in the build directory for cleaning its output
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
  else()
    message(FATAL_ERROR "Unknown Check for input file: ${CHECK_TYPE}."
      "Known checks are: execute")
  endif()

  # Double timeout if address sanitizer is enabled.
  if (ASAN)
    math(EXPR TIMEOUT "2 * ${TIMEOUT}")
  endif()

  # Multiply timeout by the user option
  # Note: "1" is parsed as "ON" by cmake
  if (NOT "${SPECTRE_INPUT_FILE_TEST_TIMEOUT_FACTOR}" STREQUAL ON)
    math(EXPR TIMEOUT "${SPECTRE_INPUT_FILE_TEST_TIMEOUT_FACTOR} * ${TIMEOUT}")
  endif()

  set_tests_properties(
    "${CTEST_NAME}"
    PROPERTIES
    FAIL_REGULAR_EXPRESSION "ERROR"
    TIMEOUT ${TIMEOUT}
    LABELS "${TAGS}"
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endfunction()

# Searches the directory INPUT_FILE_DIR for .yaml files and adds a
# test for each one. The input files must contain a line of the form:
# '# Executable: EvolveScalarWave1D'
# with the name of the executable that should be able to parse and run
# using the input file, and a line of the form:
# '# Check: parse;execute'
# OR
# '# Check:'
# If 'execute' is present then the input file will not just be parsed,
# but the simulation will be run.
function(add_input_file_tests INPUT_FILE_DIR)
  set(INPUT_FILE_LIST "")
  file(GLOB_RECURSE INPUT_FILE_LIST ${INPUT_FILE_DIR} "${INPUT_FILE_DIR}*.yaml")
  set(TIMEOUT 2)

  foreach(INPUT_FILE ${INPUT_FILE_LIST})
    file(READ ${INPUT_FILE} INPUT_FILE_CONTENTS)
    # Check if the executable name is present
    string(REGEX MATCH "#[ ]*Executable:[^\n]+"
      INPUT_FILE_EXECUTABLE "${INPUT_FILE_CONTENTS}")
    if("${INPUT_FILE_EXECUTABLE}" STREQUAL "")
      message(FATAL_ERROR "Could not find the executable in input "
        "file ${INPUT_FILE}. You must supply a line of the form:"
        "'# Executable: EXECUTABLE_NAME'")
    endif()
    # Extract executable name, and remove trailing white space
    string(REGEX REPLACE "#[ ]*Executable:[ ]*" ""
      INPUT_FILE_EXECUTABLE "${INPUT_FILE_EXECUTABLE}")
    string(STRIP "${INPUT_FILE_EXECUTABLE}" INPUT_FILE_EXECUTABLE)

    string(REGEX MATCH "#[ ]*CommandLineArgs:[^\n]+"
      COMMAND_LINE_ARGS "${INPUT_FILE_CONTENTS}")
    string(REGEX REPLACE "#[ ]*CommandLineArgs:[ ]*" ""
      COMMAND_LINE_ARGS "${COMMAND_LINE_ARGS}")
    string(STRIP "${COMMAND_LINE_ARGS}" COMMAND_LINE_ARGS)

    # Read what tests to do. Currently "execute" and "parse" are available.
    string(REGEX MATCH "#[ ]*Check:[^\n]+"
      INPUT_FILE_CHECKS "${INPUT_FILE_CONTENTS}")
    # Extract list of checks to perform
    string(REGEX REPLACE "#[ ]*Check:[ ]*" ""
      INPUT_FILE_CHECKS "${INPUT_FILE_CHECKS}")
    string(STRIP "${INPUT_FILE_CHECKS}" INPUT_FILE_CHECKS)
    set(INPUT_FILE_CHECKS "${INPUT_FILE_CHECKS}")
    list(REMOVE_DUPLICATES "INPUT_FILE_CHECKS")
    # Convert all the checks to lower case to make life easier.
    string(TOLOWER "${INPUT_FILE_CHECKS}" INPUT_FILE_CHECKS)

    # Make sure that the 'parse' check is listed. If not, print an
    # error message that explains that it's needed and why.
    list(FIND "INPUT_FILE_CHECKS" "parse" FOUND_PARSE)
    if (${FOUND_PARSE} EQUAL -1)
      message(FATAL_ERROR
        "The input file: "
        "'${INPUT_FILE}' "
        "does not specify the 'parse' check. All input file tests must"
        " specify the 'parse' check which runs the executable passing"
        " the '--check-options' flag. With this flag the executable"
        " should check that the input file parses correctly and that"
        " the values specified in the input file do not violate any"
        " bounds or sanity checks.")
    endif (${FOUND_PARSE} EQUAL -1)

    # Read the timeout duration specified in input file, empty is accepted.
    # The default duration is 2 seconds.
    string(REGEX MATCH "#[ ]*Timeout:[^\n]+"
      INPUT_FILE_TIMEOUT "${INPUT_FILE_CONTENTS}")
    if("${INPUT_FILE_TIMEOUT}" STREQUAL "")
      set(INPUT_FILE_TIMEOUT "${TIMEOUT}")
    else()
      string(REGEX REPLACE "#[ ]*Timeout:[ ]*" ""
        INPUT_FILE_TIMEOUT "${INPUT_FILE_TIMEOUT}")
      string(STRIP "${INPUT_FILE_TIMEOUT}" INPUT_FILE_TIMEOUT)
    endif()

    foreach(CHECK_TYPE ${INPUT_FILE_CHECKS})
      add_single_input_file_test(
        ${INPUT_FILE}
        ${INPUT_FILE_EXECUTABLE}
        "${COMMAND_LINE_ARGS}"
        ${CHECK_TYPE}
        ${INPUT_FILE_TIMEOUT}
        )
    endforeach()
    add_dependencies(test-executables ${INPUT_FILE_EXECUTABLE})
  endforeach()
endfunction()

# Dependencies will be added as the tests are processed.
add_custom_target(test-executables)

# Write command to execute an input file and clean its output into a shell
# script, which makes it easier to chain multiple commands
file(
  WRITE
  ${PROJECT_BINARY_DIR}/tmp/InputFileExecuteAndClean.sh
  "\
#!/bin/sh\n\
${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/CleanOutput.py -v --force \
--input-file $2 --output-dir ${CMAKE_BINARY_DIR}
${CMAKE_BINARY_DIR}/bin/$1 --input-file $2 \${3} && \
${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/CleanOutput.py -v \
--input-file $2 --output-dir ${CMAKE_BINARY_DIR}\n"
)

# Write command to execute an input file and clean its output into a shell
# script, which makes it easier to chain multiple commands
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/ExecuteCheckOutputFilesAndClean.sh
  ${PROJECT_BINARY_DIR}/tmp/ExecuteCheckOutputFilesAndClean.sh
  @ONLY)

add_input_file_tests("${CMAKE_SOURCE_DIR}/tests/InputFiles/")
