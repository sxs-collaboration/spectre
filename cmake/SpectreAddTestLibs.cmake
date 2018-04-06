# Distributed under the MIT License.
# See LICENSE.txt for details.

# The functions in this file are used to set up test libraries.
# The way to use these functions is to first set the global properties:
#   set_property(GLOBAL PROPERTY SPECTRE_TESTS_LIB_FUNCTIONS_PROPERTY "")
#   set_property(GLOBAL PROPERTY SPECTRE_TESTS_LIBS_PROPERTY "")
#
# Then have various add_subdirectory commands, normally one per library.
# Each library must compile a list of source files, which is then passed to
# add_test_library. After all the test libraries have called add_test_library
# the function "write_test_registration_function" can be called
# to write the C++ header needed for registering the tests inside all the
# libraries. Finally, the test libraries that need to be linked into the test
# runner executable can be retrieved from the global property using:
#   get_property(SPECTRE_TESTS_LIBRARIES GLOBAL PROPERTY
#                SPECTRE_TESTS_LIBS_PROPERTY)
#
# and then adding ${SPECTRE_TESTS_LIBRARIES} as a lib to target_link_libraries.
#
# Note: Both add_test_library and write_test_registration_function
# are documented below.

# We want a global variable without having to worry about scope at all since
# we provide an easy-to-use user function for adding test libraries
set_property(GLOBAL PROPERTY SPECTRE_TESTS_LIB_FUNCTIONS_PROPERTY "")
set_property(GLOBAL PROPERTY SPECTRE_TESTS_LIBS_PROPERTY "")

# Adds the library LIBRARY as testing library. The FOLDER should be the
# subdirectories of ${CMAKE_SOURCE_DIR}/tests/Unit, e.g. DataStructures
# for the DataStructures library, or Evolution/Systems/ScalarWave for
# a ScalarWave library. The source files to the library must be passed
# as the third argument, and the libraries that the test library needs
# to link to should be passed as the last argument.
function(add_test_library LIBRARY FOLDER LIBRARY_SOURCES LINK_LIBS)
  foreach (SOURCE_FILE ${LIBRARY_SOURCES})
    # We only add source files that actually call SPECTRE_TEST_CASE
    file(READ
      "${SOURCE_FILE}"
      SOURCE_FILE_CONTENTS
      )
    # Note: The result of find is -1 if not found
    string(FIND
      "${SOURCE_FILE_CONTENTS}"
      "SPECTRE_TEST_CASE("
      FOUND_SPECTRE_TEST_CASE
      )
    if(NOT ${FOUND_SPECTRE_TEST_CASE} EQUAL -1)
      # Get the name of the file without path and without extension
      get_filename_component(
        SOURCE_NAME
        ${SOURCE_FILE}
        NAME_WE
        )
      # We specify the macro SPECTRE_TEST_REGISTER_FUNCTION for each
      # source file in the library which is used by the TestingFramework.hpp
      # file to create a function that can be called from the main test
      # executable to get static variables to be initialized in the source
      # file.
      set_source_files_properties(
        ${SOURCE_FILE}
        PROPERTIES
        COMPILE_FLAGS
        "-D SPECTRE_TEST_REGISTER_FUNCTION=${LIBRARY}_${SOURCE_NAME}"
        )
      # We use the global "variable" (CMake doesn't really have global
      # variables but global properties are effectively the same)
      # SPECTRE_TESTS_LIB_FUNCTIONS_PROPERTY to keep track of all the
      # functions in all the libraries that have been added.
      # If more than one test executable exists each should add their
      # own libraries independently so that the property can be reset to
      # an empty string after the first executable's libraries were added.
      get_property(
        SPECTRE_TESTS_LIB_FUNCTIONS
        GLOBAL
        PROPERTY
        SPECTRE_TESTS_LIB_FUNCTIONS_PROPERTY)
      set(SPECTRE_TESTS_LIB_FUNCTIONS
        "${SPECTRE_TESTS_LIB_FUNCTIONS};${LIBRARY}_${SOURCE_NAME}")
      set_property(GLOBAL
        PROPERTY
        SPECTRE_TESTS_LIB_FUNCTIONS_PROPERTY
        ${SPECTRE_TESTS_LIB_FUNCTIONS})
    endif()
  endforeach()

  add_library(
    ${LIBRARY}
    ${LIBRARY_SOURCES}
    )

  target_link_libraries(
    ${LIBRARY}
    INTERFACE
    ${LINK_LIBS}
    INTERFACE
    ${PYTHON_LIBRARIES}
    )

  set_target_properties(
    ${LIBRARY}
    PROPERTIES
    FOLDER "${FOLDER}/"
    )

  target_include_directories(
    ${LIBRARY}
    SYSTEM
    PUBLIC ${NUMPY_INCLUDE_DIRS}
    PUBLIC ${PYTHON_INCLUDE_DIRS}
    )

  # We use a global variable to also make a list of all the
  # libraries added so far. Again, this is easier for users
  # than needing to propagate the libraries back up using
  # PARENT_SCOPE in various calls to set().
  get_property(
    SPECTRE_TESTS_LIBS
    GLOBAL
    PROPERTY
    SPECTRE_TESTS_LIBS_PROPERTY)
  set(SPECTRE_TESTS_LIBS
    "${SPECTRE_TESTS_LIBS};${LIBRARY}")
  set_property(GLOBAL
    PROPERTY
    SPECTRE_TESTS_LIBS_PROPERTY
    ${SPECTRE_TESTS_LIBS})
endfunction()

# Writes a header file with a function named FUNCTION_NAME that in turn
# calls all the functions in the registered source files so that tests
# in those source files are registered with Catch. The FILENAME should be
# the absolute path to where the hpp file should be written. For example,
# "${CMAKE_BINARY_DIR}/tests/Unit/RunTestsRegister.hpp"
function(write_test_registration_function
    FILENAME FUNCTION_NAME)
  get_property(
    TEST_FUNCTIONS
    GLOBAL
    PROPERTY
    SPECTRE_TESTS_LIB_FUNCTIONS_PROPERTY)
  set(REGISTERING_FUNCTIONS_DECL "")
  set(REGISTERING_FUNCTIONS_CALL "")
  foreach(TEST_FUNCTION ${TEST_FUNCTIONS})
    set(
      REGISTERING_FUNCTIONS_DECL
      "${REGISTERING_FUNCTIONS_DECL}void ${TEST_FUNCTION}() noexcept;\n")
    set(
      REGISTERING_FUNCTIONS_CALL
      "${REGISTERING_FUNCTIONS_CALL}  ${TEST_FUNCTION}();\n")
  endforeach()
  # We write the file into a tmp and then configure it so we don't
  # rebuild the tests in the case where CMake is re-run but nothing changed.
  file(WRITE
    "${FILENAME}.tmp"
    "#pragma once\n"
    "${REGISTERING_FUNCTIONS_DECL}\n"
    "inline void ${FUNCTION_NAME}() noexcept {\n"
    "${REGISTERING_FUNCTIONS_CALL}"
    "}\n"
    )
  configure_file(
    "${FILENAME}.tmp"
    "${FILENAME}"
    COPYONLY
    )
endfunction()
