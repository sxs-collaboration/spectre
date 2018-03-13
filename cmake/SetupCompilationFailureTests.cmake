# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(PythonInterp REQUIRED)

# Main function - the only one designed to be called from outside this module.
function(add_compilation_tests TEST_TARGET)
    get_target_property(SOURCE_FILES ${TEST_TARGET} SOURCES)

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tmp")

    foreach (SOURCE_FILE ${SOURCE_FILES})
      if (NOT IS_ABSOLUTE ${SOURCE_FILE})
        set(SOURCE_FILE ${CMAKE_CURRENT_LIST_DIR}/${SOURCE_FILE})
      endif()
      set(ABSOLUTE_SOURCE_FILES "${ABSOLUTE_SOURCE_FILES};${SOURCE_FILE}")
    endforeach()

    execute_process(
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/cmake/CompilationTestsParse.py
        ${ABSOLUTE_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tmp
        RESULT_VARIABLE PARSED_TESTS_SUCCESSFULLY)
    if (NOT ${PARSED_TESTS_SUCCESSFULLY} EQUAL 0)
      message(FATAL_ERROR "Failed to parse compilation test files")
    endif()

    foreach (SOURCE_FILE ${ABSOLUTE_SOURCE_FILES})
        compilation_tests_parse_file(${SOURCE_FILE} ${TEST_TARGET})
    endforeach ()
endfunction()

macro(parse_compilation_test_string)
  string(REGEX REPLACE "TAGS: ([a-z0-9_ ]+) (.*)" "\\1"
    TEST_TAGS "${OUTPUT_REGEX}")

  string(REGEX REPLACE "TAGS: ${TEST_TAGS} VERSION: ([0-9]+.[0-9]+.[0-9]+) (.*)"
    "\\1" COMPILER_VERSION "${OUTPUT_REGEX}")

  string(REGEX REPLACE
    "TAGS: ${TEST_TAGS} VERSION: ${COMPILER_VERSION} REGEX: (.*)" "\\1"
    OUTPUT_REGEX "${OUTPUT_REGEX}")

  string(FIND "${OUTPUT_REGEX}" "TAGS: " NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX)
  if (NOT -1 EQUAL ${NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX})
    string(SUBSTRING "${OUTPUT_REGEX}" ${NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX} -1
      NEXT_FULL_REGEX)
    string(FIND "${OUTPUT_REGEX}" " TAGS: " NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX)
    string(SUBSTRING "${OUTPUT_REGEX}" 0 ${NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX}
      OUTPUT_REGEX)

    while(NOT "${NEXT_FULL_REGEX}" STREQUAL "${OUTPUT_REGEX}")
      string(REGEX REPLACE "TAGS: ([a-z0-9_ ]+) VERSION: (.*)" "\\1"
        NEXT_TEST_TAGS "${NEXT_FULL_REGEX}")

      string(REGEX REPLACE
        "TAGS: ${NEXT_TEST_TAGS} VERSION: ([0-9]+.[0-9]+.[0-9]+) REGEX: (.*)"
        "\\1" NEXT_COMPILER_VERSION "${NEXT_FULL_REGEX}")

      string(REGEX REPLACE
        "TAGS: ${NEXT_TEST_TAGS} VERSION: ${NEXT_COMPILER_VERSION} REGEX: (.*)"
        "\\1" NEXT_FULL_REGEX "${NEXT_FULL_REGEX}")

      string(FIND "${NEXT_FULL_REGEX}" " TAGS: "
        NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX)
      string(SUBSTRING "${NEXT_FULL_REGEX}" 0
        ${NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX} NEXT_OUTPUT_REGEX)

      if (${NEXT_COMPILER_VERSION} VERSION_GREATER
          ${CMAKE_CXX_COMPILER_VERSION})
        break()
      endif()

      set(TEST_TAGS "${NEXT_TEST_TAGS}")
      set(COMPILER_VERSION "${NEXT_COMPILER_VERSION}")
      set(OUTPUT_REGEX "${NEXT_OUTPUT_REGEX}")

      string(FIND "${NEXT_FULL_REGEX}" "TAGS: "
        NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX)
      # If we cannot find "TAGS" anymore then we break
      if (-1 EQUAL ${NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX})
        break()
      endif()
      string(SUBSTRING "${NEXT_FULL_REGEX}"
        ${NEXT_TAGS_LOCATION_IN_OUTPUT_REGEX} -1
        NEXT_FULL_REGEX)
    endwhile()
  endif()
endmacro()


function(compilation_tests_parse_file SOURCE_FILE TEST_TARGET)
  if(NOT EXISTS ${SOURCE_FILE})
    message(WARNING
      "Could not find source file:\n\"${SOURCE_FILE}\"\nfor tests\n.")
    return()
  endif()

  file(STRINGS ${SOURCE_FILE} CONTENTS NEWLINE_CONSUME)
  string(REGEX MATCHALL
    "#ifdef COMPILATION_TEST_([^\n]+)"
    TESTS
    "${CONTENTS}")

  foreach(IFDEF_TEST_NAME ${TESTS})
    string(REGEX REPLACE "#ifdef (COMPILATION_TEST_[^\n]+)" "\\1"
      TEST_NAME "${IFDEF_TEST_NAME}")

    if(EXISTS "${CMAKE_BINARY_DIR}/tmp/${TEST_NAME}.${CMAKE_CXX_COMPILER_ID}")
      FILE(READ "${CMAKE_BINARY_DIR}/tmp/${TEST_NAME}.${CMAKE_CXX_COMPILER_ID}"
        OUTPUT_REGEX)
      FILE(REMOVE
        "${CMAKE_BINARY_DIR}/tmp/${TEST_NAME}.${CMAKE_CXX_COMPILER_ID}")

      parse_compilation_test_string()

    elseif(EXISTS "${CMAKE_BINARY_DIR}/tmp/${TEST_NAME}.all")
      FILE(READ "${CMAKE_BINARY_DIR}/tmp/${TEST_NAME}.all" OUTPUT_REGEX)
      FILE(REMOVE "${CMAKE_BINARY_DIR}/tmp/${TEST_NAME}.all")

      string(REGEX REPLACE "TAGS: ([a-z0-9_ ]+) REGEX:(.*)" "\\1"
        TEST_TAGS "${OUTPUT_REGEX}")

      string(REGEX REPLACE "TAGS: ${TEST_TAGS} REGEX: (.*)" "\\1"
        OUTPUT_REGEX "${OUTPUT_REGEX}")
    else()
      message(FATAL_ERROR "Could not find a regex to match for test: ${TEST_NAME}")
    endif()
    string(REGEX REPLACE " " ";" TEST_TAGS "${TEST_TAGS}")

    if ("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles")
      add_test(
        NAME "${TEST_NAME}"
        COMMAND make WHICH_TEST="-D${TEST_NAME}" ${TEST_TARGET}
        )
      set_tests_properties(
        "${TEST_NAME}"
        PROPERTIES
        TIMEOUT 10
        LABELS "${TEST_TAGS}"
        PASS_REGULAR_EXPRESSION ${OUTPUT_REGEX}
        )
    endif()
  endforeach()
endfunction()
