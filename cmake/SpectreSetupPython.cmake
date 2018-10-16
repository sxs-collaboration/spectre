# Distributed under the MIT License.
# See LICENSE.txt for details.

# In order to avoid runtime errors about missing CmiPrintf and other Cmi
# (Charm++) functions, we need to link in the whole PyBindings archive.
# In order to make it easier for users, we define the variable
# SPECTRE_LINK_PYBINDINGS so that the target_link_libraries only needs to
# specify ${SPECTRE_LINK_PYBINDINGS}. Note that ${SPECTRE_LINK_PYBINDINGS}
# must be the last library to link.
set(SPECTRE_LINK_PYBINDINGS
  -Wl,--whole-archive
  PUBLIC PyBindings
  PUBLIC ${HDF5_LIBRARIES}
  -Wl,--no-whole-archive)

set(SPECTRE_PYTHON_PREFIX "${CMAKE_BINARY_DIR}/bin/python/spectre/")
get_filename_component(
  SPECTRE_PYTHON_PREFIX
  "${SPECTRE_PYTHON_PREFIX}" ABSOLUTE)
get_filename_component(
  SPECTRE_PYTHON_PREFIX_PARENT
  "${SPECTRE_PYTHON_PREFIX}/.." ABSOLUTE)

# Create the root __init__.py file
if(NOT EXISTS "${SPECTRE_PYTHON_PREFIX}/__init__.py")
  file(WRITE
    "${SPECTRE_PYTHON_PREFIX}/__init__.py"
    "__all__ = []")
endif()

# Write a file to be able to set up the new python path.
file(WRITE
  "${CMAKE_BINARY_DIR}/tmp/LoadPython.sh"
  "#!/bin/sh\n"
  "export PYTHONPATH=$PYTHONPATH:${SPECTRE_PYTHON_PREFIX_PARENT}")
configure_file(
  "${CMAKE_BINARY_DIR}/tmp/LoadPython.sh"
  "${CMAKE_BINARY_DIR}/bin/LoadPython.sh")

# Add a python module, either with or without python bindings and with
# or without additional python files. If bindings are being provided then
# the library will be named Py${MODULE_NAME}, e.g. if MODULE_NAME is
# DataStructures then the library name is PyDataStructures.
#
# - MODULE_NAME   The name of the module, e.g. DataStructures.
#
# - MODULE_PATH   Path inside the module, e.g. submodule0/submodule1 would
#                 result in loading spectre.submodule0.submodule1
#
# - SOURCES       The C++ source files for bindings. Omit if no bindings
#                 are being generated.
#
# - PYTHON_FILES  List of the python files (relative to
#                 ${CMAKE_SOURCE_DIR}/src) to add to the module. Omit if
#                 no python files are to be provided.
function(SPECTRE_PYTHON_ADD_MODULE MODULE_NAME)
  set(SINGLE_VALUE_ARGS MODULE_PATH)
  set(MULTI_VALUE_ARGS SOURCES PYTHON_FILES)
  cmake_parse_arguments(
    ARG ""
    "${SINGLE_VALUE_ARGS}"
    "${MULTI_VALUE_ARGS}"
    ${ARGN})

  set(MODULE_LOCATION
    "${SPECTRE_PYTHON_PREFIX}/${ARG_MODULE_PATH}/${MODULE_NAME}")
  get_filename_component(MODULE_LOCATION ${MODULE_LOCATION} ABSOLUTE)

  # Create list of all the python submodule names
  set(PYTHON_SUBMODULES "")
  foreach(PYTHON_FILE ${ARG_PYTHON_FILES})
    # Get file name Without Extension (NAME_WE)
    get_filename_component(PYTHON_FILE "${PYTHON_FILE}" NAME_WE)
    list(APPEND PYTHON_SUBMODULES ${PYTHON_FILE})
  endforeach(PYTHON_FILE ${ARG_PYTHON_FILES})

  # Add our python library, if it has sources
  set(SPECTRE_PYTHON_MODULE_IMPORT "")
  if(NOT "${ARG_SOURCES}" STREQUAL "")
    add_library("Py${MODULE_NAME}" MODULE ${ARG_SOURCES})
    # We don't want the 'lib' prefix for python modules, so we set the output name
    SET_TARGET_PROPERTIES(
      "Py${MODULE_NAME}"
      PROPERTIES
      PREFIX ""
      LIBRARY_OUTPUT_NAME "_${MODULE_NAME}"
      LIBRARY_OUTPUT_DIRECTORY ${MODULE_LOCATION}
      )
    set(SPECTRE_PYTHON_MODULE_IMPORT "from ._${MODULE_NAME} import *")
    add_dependencies(test-executables "Py${MODULE_NAME}")
  endif(NOT "${ARG_SOURCES}" STREQUAL "")

  # Read the __init__.py file if it exists
  set(INIT_FILE_LOCATION "${MODULE_LOCATION}/__init__.py")
  set(INIT_FILE_CONTENTS "")
  if(EXISTS ${INIT_FILE_LOCATION})
    file(READ
      ${INIT_FILE_LOCATION}
      INIT_FILE_CONTENTS)
  endif(EXISTS ${INIT_FILE_LOCATION})

  # Update the "from ._LIB import *" in the __init__.py
  if("${INIT_FILE_CONTENTS}" STREQUAL "")
    set(INIT_FILE_OUTPUT "${SPECTRE_PYTHON_MODULE_IMPORT}\n__all__ = []\n")
  else("${INIT_FILE_CONTENTS}" STREQUAL "")
    string(REGEX REPLACE
      "from[^\n]+"
      "${SPECTRE_PYTHON_MODULE_IMPORT}"
      INIT_FILE_OUTPUT
      ${INIT_FILE_CONTENTS})
  endif("${INIT_FILE_CONTENTS}" STREQUAL "")

  # configure the source files into the build directory and make sure
  # they are in the init file.
  foreach(PYTHON_FILE ${ARG_PYTHON_FILES})
    # Configure file
    get_filename_component(PYTHON_FILE_JUST_NAME
      "${CMAKE_SOURCE_DIR}/src/${PYTHON_FILE}" NAME)
    configure_file(
      "${CMAKE_SOURCE_DIR}/src/${PYTHON_FILE}"
      "${MODULE_LOCATION}/${PYTHON_FILE_JUST_NAME}"
      )

    # Update init file
    get_filename_component(PYTHON_FILE_JUST_NAME_WE
      "${CMAKE_SOURCE_DIR}/src/${PYTHON_FILE}" NAME_WE)
    string(FIND
      ${INIT_FILE_OUTPUT}
      "\"${PYTHON_FILE_JUST_NAME_WE}\""
      INIT_FILE_CONTAINS_ME)
    if(${INIT_FILE_CONTAINS_ME} EQUAL -1)
      string(REPLACE
        "__all__ = ["
        "__all__ = [\"${PYTHON_FILE_JUST_NAME_WE}\", "
        INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})
    endif(${INIT_FILE_CONTAINS_ME} EQUAL -1)
  endforeach(PYTHON_FILE ${ARG_PYTHON_FILES})

  string(REPLACE ", ]" "]" INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})

  # Remove python files that we are no longer using
  string(REGEX MATCH "\\\[.*\\\]"
    WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS ${INIT_FILE_OUTPUT})
  string(REPLACE "[" "" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  string(REPLACE "]" "" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  string(REPLACE "\"" "" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  string(REPLACE ", " ";" WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS
    "${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS}")
  foreach(CURRENT_PYTHON_MODULE in ${WRITTEN_PYTHON_ALL_WITHOUT_EXTENSIONS})
    string(FIND "${ARG_PYTHON_FILES}" "${CURRENT_PYTHON_MODULE}.py"
      PYTHON_MODULE_EXISTS)
    if(${PYTHON_MODULE_EXISTS} EQUAL -1)
      string(REPLACE "\"${CURRENT_PYTHON_MODULE}\""
        ""
        INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})
      string(REPLACE ", ," "," INIT_FILE_OUTPUT ${INIT_FILE_OUTPUT})
      file(REMOVE "${MODULE_LOCATION}/${CURRENT_PYTHON_MODULE}.py")
    endif(${PYTHON_MODULE_EXISTS} EQUAL -1)
  endforeach(CURRENT_PYTHON_MODULE in ${MATCHED})

  # Write the __init__.py file for the module
  if(NOT ${INIT_FILE_OUTPUT} STREQUAL "${INIT_FILE_CONTENTS}")
    file(WRITE ${INIT_FILE_LOCATION} ${INIT_FILE_OUTPUT})
  endif(NOT ${INIT_FILE_OUTPUT} STREQUAL "${INIT_FILE_CONTENTS}")

  # Register with parent submodules:
  # We walk up the tree until we get to ${SPECTRE_PYTHON_PREFIX}
  # and make sure we have all the submodules registered.
  set(CURRENT_MODULE ${MODULE_LOCATION})
  while(NOT ${CURRENT_MODULE} STREQUAL ${SPECTRE_PYTHON_PREFIX})
    get_filename_component(PARENT_MODULE "${CURRENT_MODULE}/.." ABSOLUTE)
    string(REPLACE "${PARENT_MODULE}/" ""
      CURRENT_MODULE_NAME ${CURRENT_MODULE})

    set(PARENT_MODULE_CONTENTS "__all__ = []")
    if(EXISTS "${PARENT_MODULE}/__init__.py")
      file(READ
        "${PARENT_MODULE}/__init__.py"
        PARENT_MODULE_CONTENTS)
    endif(EXISTS "${PARENT_MODULE}/__init__.py")

    string(FIND "${PARENT_MODULE_CONTENTS}" "\"${CURRENT_MODULE_NAME}\""
      PARENT_MODULE_CONTAINS_ME)

    if(${PARENT_MODULE_CONTAINS_ME} EQUAL -1)
      string(REPLACE "__all__ = [" "__all__ = [\"${CURRENT_MODULE_NAME}\", "
        PARENT_MODULE_NEW_CONTENTS "${PARENT_MODULE_CONTENTS}")
      string(REPLACE ", ]" "]"
        PARENT_MODULE_NEW_CONTENTS "${PARENT_MODULE_NEW_CONTENTS}")
      file(WRITE
        "${PARENT_MODULE}/__init__.py"
        ${PARENT_MODULE_NEW_CONTENTS})
    endif(${PARENT_MODULE_CONTAINS_ME} EQUAL -1)

    set(CURRENT_MODULE ${PARENT_MODULE})
  endwhile(NOT ${CURRENT_MODULE} STREQUAL ${SPECTRE_PYTHON_PREFIX})
endfunction()

# Register a python test file with ctest.
# - TEST_NAME    The name of the test,
#                e.g. "Unit.DataStructures.Python.DataVector"
#
# - FILE         The file to add, e.g. Test_DataVector.py
#
# - TAGS         A semicolon separated list of labels for the test,
#                e.g. "unit;DataStructures;python"
function(SPECTRE_ADD_PYTHON_TEST TEST_NAME FILE TAGS)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
  get_filename_component(FILE "${FILE}" ABSOLUTE)
  string(TOLOWER "${TAGS}" TAGS)

  add_test(
    NAME "\"${TEST_NAME}\""
    COMMAND
    ${PYTHON_EXECUTABLE}
    ${FILE}
    )

  # The fail regular expression is what Python.unittest returns when no
  # tests are found to be run. We treat this as a test failure.
  set_tests_properties(
    "\"${TEST_NAME}\""
    PROPERTIES
    FAIL_REGULAR_EXPRESSION "Ran 0 test"
    TIMEOUT 2
    LABELS "${TAGS}"
    ENVIRONMENT "PYTHONPATH=${SPECTRE_PYTHON_PREFIX_PARENT}:\$PYTHONPATH"
    )
endfunction()
