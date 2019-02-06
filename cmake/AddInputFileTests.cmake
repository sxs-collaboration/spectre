# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(PythonInterp REQUIRED)

function(add_single_input_file_test INPUT_FILE EXECUTABLE CHECK_TYPE TIMEOUT)
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
      ${EXECUTABLE} ${INPUT_FILE}
      # Make sure we run the test in the build directory for cleaning its output
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
  else()
    message(FATAL_ERROR "Unknown Check for input file: ${CHECK_TYPE}."
      "Known checks are: execute")
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
# '# Check: execute'
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

    # Read what tests to do. Currently only "execute" is available
    # "parse" is ignored because it's always run, empty is accepted.
    string(REGEX MATCH "#[ ]*Check:[^\n]+"
      INPUT_FILE_CHECKS "${INPUT_FILE_CONTENTS}")
    # Extract list of checks to perform
    string(REGEX REPLACE "#[ ]*Check:[ ]*" ""
      INPUT_FILE_CHECKS "${INPUT_FILE_CHECKS}")
    string(STRIP "${INPUT_FILE_CHECKS}" INPUT_FILE_CHECKS)
    set(INPUT_FILE_CHECKS "parse;${INPUT_FILE_CHECKS}")
    list(REMOVE_DUPLICATES "INPUT_FILE_CHECKS")

    foreach(CHECK_TYPE ${INPUT_FILE_CHECKS})
      add_single_input_file_test(
        ${INPUT_FILE}
        ${INPUT_FILE_EXECUTABLE}
        ${CHECK_TYPE}
        ${TIMEOUT}
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
${CMAKE_BINARY_DIR}/bin/$1 --input-file $2\n\
execute_exit=$?\n\
${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/CleanOutput.py -v \
--input-file $2 --output-dir ${CMAKE_BINARY_DIR} && exit \${execute_exit}\n"
)

add_input_file_tests("${CMAKE_SOURCE_DIR}/tests/InputFiles/")
