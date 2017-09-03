# Distributed under the MIT License.
# See LICENSE.txt for details.
#
# This file contains functions used by CMake to add Catch tests to CTest.
#
# Catch is a testing framework available on Github:
# https://github.com/philsquared/Catch
# It supports a variety of different styles of tests including BDD and fixture
# tests.
#
# Usage
# =====
# To run the tests, type 'ctest' in the build directory. You can specify
# a regex to match the test name using 'ctest -R Unit.Blah', or run all
# tests with a certain tag using 'ctest -L tag'
#
# Attributes
# ==========
# Attributes allow you to modify properties of the test. Attributes are
# specified as follows:
# // [[TimeOut, 10]]
# // [[OutputRegex, A regular expression that is expected to match the output
# //   from the test]]
# SPECTRE_TEST_CASE("Unit.Blah", "[Unit]") {
#
# Note the space after the comma!
#
# Available attributes are:
# TimeOut - override the default timeout and set the timeout to N seconds. This
#           should be set very sparingly since unit tests are designed to be
#           short. If your test is too long you should consider testing smaller
#           portions of the code if possible, or writing an integration test
#           instead.
# OutputRegex - When testing failure modes the exact error message must be
#               tested, not just that the test failed. Since the string passed
#               is a regular expression you must escape any regex tokens. For
#               example, to match "some (word) and" you must specify the
#               string "some \(word\) and".

find_package(PythonInterp REQUIRED)

# Main function - the only one designed to be called from outside this module.
function(spectre_add_catch_tests TEST_TARGET)
    get_target_property(SOURCE_FILES ${TEST_TARGET} SOURCES)
    # For each of the source files, we use spectre_parse_file to find all the
    # Catch tests inside the source file and add them to CTest.
    # We ignore the Charm++ generated header files.

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tmp")

    foreach (SOURCE_FILE ${SOURCE_FILES})
        string(REGEX MATCH ".*(decl.h|def.h)" CHARM_INTERFACE_FILE
               "${SOURCE_FILE}")
        if (NOT CHARM_INTERFACE_FILE)
            if (NOT IS_ABSOLUTE ${SOURCE_FILE})
                set(SOURCE_FILE ${CMAKE_CURRENT_LIST_DIR}/${SOURCE_FILE})
            endif()
            set(ABSOLUTE_SOURCE_FILES "${ABSOLUTE_SOURCE_FILES};${SOURCE_FILE}")
        endif()
    endforeach()

    execute_process(
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/cmake/SpectreParseTests.py
        ${ABSOLUTE_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tmp
        RESULT_VARIABLE PARSED_TESTS_SUCCESSFULLY)
    if (${PARSED_TESTS_SUCCESSFULLY} GREATER 0)
      message(FATAL_ERROR "Failed to parse test files")
    endif()

    foreach (SOURCE_FILE ${ABSOLUTE_SOURCE_FILES})
        spectre_parse_file(${SOURCE_FILE} ${TEST_TARGET})
    endforeach ()
endfunction()

# Parses the cpp file and extracts the tests specified in it. Each test is then
# added to CTest, and the run command is adjusted according to timeout,
# willfail, and serialization needs.
function(spectre_parse_file SOURCE_FILE TEST_TARGET)
    if (NOT EXISTS ${SOURCE_FILE})
        message(WARNING "Could not find source file:\n\"${SOURCE_FILE}\"\nfor \
        tests\n.")
        return()
    endif ()

    file(STRINGS ${SOURCE_FILE} CONTENTS NEWLINE_CONSUME)

    # Remove commented out tests so they are not added to ctest
    string(REGEX REPLACE "\n[ \t]*//+[^\n]+" "\n" CONTENTS "${CONTENTS}")

    # The regex searches for SPECTRE_TEST_CASE_METHOD, SCENARIO and SPECTRE_TEST_CASE
    # to find tests that need to be added.  TESTS will be a list of all tests
    # found in the SOURCE_FILE.
    string(REGEX MATCHALL
           "(CATCH_)?(SPECTRE_TEST_CASE_METHOD|SCENARIO|SPECTRE_TEST_CASE)[ \t]*\\([^\)]+\\)[ \t]*{"
           TESTS
           "${CONTENTS}")

    foreach (TEST_NAME ${TESTS})
        # Get test type and fixture if applicable
        string(REGEX MATCH "(CATCH_)?(SPECTRE_TEST_CASE_METHOD|SCENARIO|SPECTRE_TEST_CASE)"
               TEST_TYPE "${TEST_NAME}")

        string(REPLACE "${TEST_TYPE}(" ""
               TEST_FIXTURE "${TEST_TYPE_AND_FIXTURE}")

        # Get string parts of test definition
        string(REGEX MATCHALL "\"[^\"]+\"" TEST_STRINGS "${TEST_NAME}")

        # Strip wrapping quotation marks of each element of the list
        # TEST_STRINGS
        string(REGEX REPLACE "^\"(.*)\"$" "\\1" TEST_STRINGS "${TEST_STRINGS}")
        string(REPLACE "\";\"" ";" TEST_STRINGS "${TEST_STRINGS}")

        # Validate that a test name and tags have been provided
        list(LENGTH TEST_STRINGS TEST_STRINGS_LENGTH)
        if (NOT TEST_STRINGS_LENGTH EQUAL 2)
            message(FATAL_ERROR
                    "You must provide a valid test name and tags "
                    "for all tests in ${SOURCE_FILE}. Cannot use the test:\n"
                    "\"${TEST_STRINGS}\"\n")
        endif ()

        # Assign name and tags
        list(GET TEST_STRINGS 0 NAME)
        if ("${TEST_TYPE}" STREQUAL "SCENARIO")
            set(NAME "Scenario: ${NAME}")
        endif ()
        set(CTEST_NAME "${NAME}")

        # Gets the TAGS of the test which is element 1 of TEST_STRINGS,
        # strips the enclosing brackets, and makes a list
        list(GET TEST_STRINGS 1 TAGS)
        string(TOLOWER "${TAGS}" TAGS)
        string(REPLACE "]" ";" TAGS "${TAGS}")
        string(REPLACE "[" "" TAGS "${TAGS}")

        # These files are generated by the SpectreParseTests.py
        file(READ "${CMAKE_BINARY_DIR}/tmp/${CTEST_NAME}.output_regex" OUTPUT_REGEX)
        file(REMOVE "${CMAKE_BINARY_DIR}/tmp/${CTEST_NAME}.output_regex")
        file(READ "${CMAKE_BINARY_DIR}/tmp/${CTEST_NAME}.timeout" TIMEOUT)
        file(REMOVE "${CMAKE_BINARY_DIR}/tmp/${CTEST_NAME}.timeout")

        # The default TIMEOUT is set to -1, but overwritten by each type
        # of test or a specified TIMEOUT attribute for a given test.
        # Thus we can use it to check if each test has been tagged by an
        # appropriate type.
        if (TIMEOUT EQUAL -1)
            message(FATAL_ERROR
                    "You must set at least one tag of value [unit] "
                    "for \"${NAME}\" in ${SOURCE_FILE}\n")
        endif ()

        # Add the test and set its properties
        add_test(NAME "\"${CTEST_NAME}\""
                 COMMAND $<TARGET_FILE:${TEST_TARGET}>
                 \"${NAME}\" --durations yes
                 --warn NoAssertions
                 --name "\"$<CONFIGURATION>.${CTEST_NAME}\"")

        # Check if the test is supposed to fail. If so then let ctest know
        # that a failed test is actually a pass.
        if (NOT "${OUTPUT_REGEX}" STREQUAL "None")
            if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                set(OUTPUT_REGEX
                    "${OUTPUT_REGEX}|### No ASSERT tests in release mode ###")
            endif()
            set_tests_properties(
                "\"${CTEST_NAME}\"" PROPERTIES
                FAIL_REGULAR_EXPRESSION "No tests ran"
                TIMEOUT ${TIMEOUT}
                PASS_REGULAR_EXPRESSION "${OUTPUT_REGEX}"
                LABELS "${TAGS}"
                ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
        else ()
            set_tests_properties(
                "\"${CTEST_NAME}\"" PROPERTIES
                FAIL_REGULAR_EXPRESSION "No tests ran"
                TIMEOUT ${TIMEOUT}
                LABELS "${TAGS}"
                ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
        endif ()
    endforeach ()
endfunction()
