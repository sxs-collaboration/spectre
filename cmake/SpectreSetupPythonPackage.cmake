# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_define_test_timeout_factor_option(PYTHON "Python")

option(PY_DEV_MODE "Enable development mode for the Python package, meaning \
that Python files are symlinked rather than copied to the build directory" OFF)
function(configure_or_symlink_py_file SOURCE_FILE TARGET_FILE)
  if(PY_DEV_MODE)
    get_filename_component(_TARGET_FILE_DIR ${TARGET_FILE} DIRECTORY)
    execute_process(COMMAND
      ${CMAKE_COMMAND} -E make_directory ${_TARGET_FILE_DIR})
    execute_process(COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${SOURCE_FILE} ${TARGET_FILE})
  else()
    configure_file(${SOURCE_FILE} ${TARGET_FILE})
  endif()
endfunction()

set(SPECTRE_PYTHON_PREFIX "${SPECTRE_PYTHON_PREFIX_PARENT}/spectre")

# Create the root __init__.py file
configure_or_symlink_py_file(
  "${CMAKE_SOURCE_DIR}/support/Python/__init__.py"
  "${SPECTRE_PYTHON_PREFIX}/__init__.py"
)

# Create the root __main__.py entry point
configure_or_symlink_py_file(
  "${CMAKE_SOURCE_DIR}/support/Python/__main__.py"
  "${SPECTRE_PYTHON_PREFIX}/__main__.py"
)
# Also link the main entry point to bin/
set(PYTHON_EXE_COMMAND "-m spectre")
set(JEMALLOC_PRELOAD "")
if(BUILD_PYTHON_BINDINGS AND "${JEMALLOC_LIB_TYPE}" STREQUAL SHARED)
  set(JEMALLOC_PRELOAD "LD_PRELOAD=\${LD_PRELOAD}\${LD_PRELOAD:+:}${JEMALLOC_LIBRARIES}")
endif()
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/SpectrePythonExecutable.sh"
  "${CMAKE_BINARY_DIR}/tmp/spectre")
file(COPY "${CMAKE_BINARY_DIR}/tmp/spectre"
  DESTINATION "${CMAKE_BINARY_DIR}/bin"
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
    GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Also link the Python interpreter to bin/python-spectre as an easy way to jump
# into a Python shell or run a script that uses pybindings
set(PYTHON_EXE_COMMAND "")
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/SpectrePythonExecutable.sh"
  "${CMAKE_BINARY_DIR}/tmp/python-spectre")
file(COPY "${CMAKE_BINARY_DIR}/tmp/python-spectre"
  DESTINATION "${CMAKE_BINARY_DIR}/bin"
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
    GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Write configuration files for installing the Python modules
file(STRINGS
  ${CMAKE_SOURCE_DIR}/support/Python/requirements.txt
  SPECTRE_PY_DEPS)
file(STRINGS
  ${CMAKE_SOURCE_DIR}/support/Python/dev_requirements.txt
  SPECTRE_PY_DEV_DEPS)
list(FILTER SPECTRE_PY_DEPS EXCLUDE REGEX "^#")
list(FILTER SPECTRE_PY_DEV_DEPS EXCLUDE REGEX "^#")
list(REMOVE_ITEM SPECTRE_PY_DEPS "")
list(REMOVE_ITEM SPECTRE_PY_DEV_DEPS "")
list(JOIN SPECTRE_PY_DEPS "\n    " SPECTRE_PY_DEPS_OUTPUT)
list(JOIN SPECTRE_PY_DEV_DEPS "\n    " SPECTRE_PY_DEV_DEPS_OUTPUT)
configure_or_symlink_py_file(
  "${CMAKE_SOURCE_DIR}/pyproject.toml"
  "${SPECTRE_PYTHON_PREFIX_PARENT}/pyproject.toml")
configure_or_symlink_py_file(
  "${CMAKE_SOURCE_DIR}/setup.cfg"
  "${SPECTRE_PYTHON_PREFIX_PARENT}/setup.cfg")

set(_JEMALLOC_MESSAGE "")
if(BUILD_PYTHON_BINDINGS AND "${JEMALLOC_LIB_TYPE}" STREQUAL SHARED)
  set(_JEMALLOC_MESSAGE
    "echo 'You must run python as:'\n"
    "echo 'LD_PRELOAD=${JEMALLOC_LIBRARIES} python ...'\n")
  string(REPLACE ";" "" _JEMALLOC_MESSAGE "${_JEMALLOC_MESSAGE}")
endif()

# Write a file to be able to set up the new python path.
file(WRITE
  "${CMAKE_BINARY_DIR}/tmp/LoadPython.sh"
  "#!/bin/sh\n"
  "export PYTHONPATH=${PYTHONPATH}\n"
  ${_JEMALLOC_MESSAGE}
  )
configure_file(
  "${CMAKE_BINARY_DIR}/tmp/LoadPython.sh"
  "${CMAKE_BINARY_DIR}/bin/LoadPython.sh")

# Install the SpECTRE Python package to the CMAKE_INSTALL_PREFIX, using pip.
# This will install the package into the expected subdirectory, typically
# `lib/pythonX.Y/site-packages/`. It also creates symlinks to entry points
# specified in `setup.py`.
install(
  CODE "execute_process(\
    COMMAND ${Python_EXECUTABLE} -m pip install \
      --no-deps --no-input --no-cache-dir --no-index --ignore-installed \
      --disable-pip-version-check --no-build-isolation \
      --prefix ${CMAKE_INSTALL_PREFIX} ${SPECTRE_PYTHON_PREFIX_PARENT} \
    )"
  )

add_custom_target(all-pybindings)

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
# - LIBRARY_NAME  The name of the C++ libray, e.g. PyDataStructures.
#                 Required if SOURCES are specified. Must begin with "Py".
#
# - PYTHON_FILES  List of the python files (relative to
#                 ${CMAKE_SOURCE_DIR}/src) to add to the module. Omit if
#                 no python files are to be provided.
function(SPECTRE_PYTHON_ADD_MODULE MODULE_NAME)
  if(BUILD_PYTHON_BINDINGS AND
      "${JEMALLOC_LIB_TYPE}" STREQUAL STATIC
      AND BUILD_SHARED_LIBS)
    message(FATAL_ERROR
      "Cannot build python bindings when using a static library JEMALLOC and "
      "building SpECTRE with shared libraries. Either disable the python "
      "bindings using -D BUILD_PYTHON_BINDINGS=OFF, switch to a shared/dynamic "
      "JEMALLOC library, use the system allocator by passing "
      "-D MEMORY_ALLOCATOR=SYSTEM to CMake, or build SpECTRE using static "
      "libraries by passing -D BUILD_SHARED_LIBS=OFF to CMake.")
  endif()

  set(SINGLE_VALUE_ARGS MODULE_PATH LIBRARY_NAME)
  set(MULTI_VALUE_ARGS SOURCES PYTHON_FILES)
  cmake_parse_arguments(
    ARG ""
    "${SINGLE_VALUE_ARGS}"
    "${MULTI_VALUE_ARGS}"
    ${ARGN})

  set(MODULE_LOCATION
    "${SPECTRE_PYTHON_PREFIX}/${ARG_MODULE_PATH}/${MODULE_NAME}")
  get_filename_component(MODULE_LOCATION ${MODULE_LOCATION} ABSOLUTE)

  # Add our python library, if it has sources
  if(BUILD_PYTHON_BINDINGS AND NOT "${ARG_SOURCES}" STREQUAL "")
    if("${ARG_LIBRARY_NAME}" STREQUAL "")
      message(FATAL_ERROR "Set a LIBRARY_NAME for Python module "
          "'${MODULE_NAME}' that has sources.")
    endif()
    if(NOT "${ARG_LIBRARY_NAME}" MATCHES "^Py")
      message(FATAL_ERROR "The LIBRARY_NAME for Python module "
          "'${MODULE_NAME}' must begin with 'Py' but is '${ARG_LIBRARY_NAME}'.")
    endif()

    Python_add_library(${ARG_LIBRARY_NAME} MODULE ${ARG_SOURCES})
    set_target_properties(
      ${ARG_LIBRARY_NAME}
      PROPERTIES
      # These can be turned on once we support them
      INTERPROCEDURAL_OPTIMIZATION OFF
      CXX__VISIBILITY_PRESET OFF
      VISIBLITY_INLINES_HIDDEN OFF
      )
    # In order to avoid runtime errors about missing compatibility functions
    # defined in the PythonBindings library, we need to link in the whole
    # archive. This is not needed on macOS.
    if (APPLE)
      target_link_libraries(
        ${ARG_LIBRARY_NAME}
        PUBLIC PythonBindings
        )
    else()
      target_link_libraries(
        ${ARG_LIBRARY_NAME}
        PUBLIC
        -Wl,--whole-archive
        PythonBindings
        -Wl,--no-whole-archive
        )
    endif()
    target_link_libraries(
      ${ARG_LIBRARY_NAME}
      PRIVATE
      CharmModuleInit
      SpectreFlags
      )
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      # Clang doesn't by default enable sized deallocation so we need to
      # enable it explicitly. This can potentially cause problems if the
      # standard library being used is too old, but GCC doesn't have any
      # safeguards against that either.
      #
      # See https://github.com/pybind/pybind11/issues/1604
      target_compile_options(${ARG_LIBRARY_NAME}
        PRIVATE -fsized-deallocation)
    endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # We don't want the 'lib' prefix for python modules, so we set the output
    # name
    set_target_properties(
      ${ARG_LIBRARY_NAME}
      PROPERTIES
      PREFIX ""
      LIBRARY_OUTPUT_NAME "_Pybindings"
      LIBRARY_OUTPUT_DIRECTORY ${MODULE_LOCATION}
      )
    # We need --no-as-needed since each python module needs to depend on all the
    # shared libraries in order to run successfully.
    set(PY_LIB_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS}")
    if(NOT APPLE)
      set(PY_LIB_LINK_FLAGS
        "${CMAKE_CXX_LINK_FLAGS} -Wl,--no-as-needed")
    endif()
    set_target_properties(
      ${ARG_LIBRARY_NAME}
      PROPERTIES
      LINK_FLAGS "${PY_LIB_LINK_FLAGS}"
      )
    if(BUILD_TESTING)
      add_dependencies(test-executables ${ARG_LIBRARY_NAME})
    endif()
    add_dependencies(all-pybindings ${ARG_LIBRARY_NAME})
  endif(BUILD_PYTHON_BINDINGS AND NOT "${ARG_SOURCES}" STREQUAL "")

  # configure the Python source files into the build directory
  foreach(PYTHON_FILE ${ARG_PYTHON_FILES})
    # Configure file
    get_filename_component(PYTHON_FILE_JUST_NAME
      "${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_FILE}" NAME)
    configure_or_symlink_py_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_FILE}"
      "${MODULE_LOCATION}/${PYTHON_FILE_JUST_NAME}"
      )
  endforeach(PYTHON_FILE ${ARG_PYTHON_FILES})

  # Create empty __init__.py files if none exist
  # We walk up the tree until we get to ${SPECTRE_PYTHON_PREFIX}
  set(CURRENT_MODULE ${MODULE_LOCATION})
  while(NOT ${CURRENT_MODULE} STREQUAL ${SPECTRE_PYTHON_PREFIX})
    set(INIT_FILE_LOCATION "${CURRENT_MODULE}/__init__.py")
    if(NOT EXISTS ${INIT_FILE_LOCATION})
      file(WRITE ${INIT_FILE_LOCATION} "")
    endif()
    get_filename_component(CURRENT_MODULE "${CURRENT_MODULE}/.." ABSOLUTE)
  endwhile()
endfunction()

# Add headers if Python bindings are being built
function (spectre_python_headers LIBRARY_NAME)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
spectre_target_headers(
    ${LIBRARY_NAME}
    # Forward all remaining arguments
    ${ARGN}
    )
endfunction()

# Link with the LIBRARIES if Python bindings are being built
function (spectre_python_link_libraries LIBRARY_NAME)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
  target_link_libraries(
    ${LIBRARY_NAME}
    # Forward all remaining arguments
    ${ARGN}
    )
endfunction()

# Add the DEPENDENCIES if Python bindings are being built
function (spectre_python_add_dependencies LIBRARY_NAME)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
  add_dependencies(
    ${LIBRARY_NAME}
    # Forward all remaining arguments
    ${ARGN}
    )
endfunction()

# Register a python test file with ctest.
# - TEST_NAME    The name of the test,
#                e.g. "Unit.DataStructures.Python.DataVector"
#
# - FILE         The file to add, e.g. Test_DataVector.py
#
# - TAGS         A semicolon separated list of labels for the test,
#                e.g. "Unit;DataStructures;Python"
# - PY_MODULE_DEPENDENCY
#                The python module that this test depends on
#                Set to None if there is no python module dependency
function(SPECTRE_ADD_PYTHON_TEST TEST_NAME FILE TAGS
    PY_MODULE_DEPENDENCY)
  get_filename_component(FILE "${FILE}" ABSOLUTE)
  string(TOLOWER "${TAGS}" TAGS)

  add_test(
    NAME "${TEST_NAME}"
    COMMAND
    ${Python_EXECUTABLE}
    ${FILE}
    )

  spectre_test_timeout(TIMEOUT PYTHON 2)

  set(_TEST_ENV_VARS "PYTHONPATH=${PYTHONPATH}")
  if(BUILD_PYTHON_BINDINGS AND
      "${JEMALLOC_LIB_TYPE}" STREQUAL SHARED)
    list(APPEND
      _TEST_ENV_VARS
      "LD_PRELOAD=${JEMALLOC_LIBRARIES}"
      )
  endif()

  # The fail regular expression is what Python.unittest returns when no
  # tests are found to be run. We treat this as a test failure.
  set_tests_properties(
    "${TEST_NAME}"
    PROPERTIES
    FAIL_REGULAR_EXPRESSION "Ran 0 test"
    TIMEOUT ${TIMEOUT}
    LABELS "${TAGS};Python"
    ENVIRONMENT "${_TEST_ENV_VARS}"
    )
  # check if this is a unit test, and if so add it to the dependencies
  foreach(LABEL ${TAGS})
    string(TOLOWER "${LABEL}" LOWER_LABEL)
    string(TOLOWER "${PY_MODULE_DEPENDENCY}" LOWER_DEP)
    if("${LOWER_LABEL}" STREQUAL "unit"
        AND NOT "${LOWER_DEP}" STREQUAL "none")
      add_dependencies(unit-tests ${PY_MODULE_DEPENDENCY})
    endif()
  endforeach()
endfunction()

# Register a python test file that uses bindings with ctest.
# - TEST_NAME    The name of the test,
#                e.g. "Unit.DataStructures.Python.DataVector"
#
# - FILE         The file to add, e.g. Test_DataVector.py
#
# - TAGS         A semicolon separated list of labels for the test,
#                e.g. "Unit;DataStructures;Python"
# - PY_MODULE_DEPENDENCY
#                The python module that this test depends on
#                Set to None if there is no python module dependency
function(SPECTRE_ADD_PYTHON_BINDINGS_TEST TEST_NAME FILE TAGS
    PY_MODULE_DEPENDENCY)
  if(NOT BUILD_PYTHON_BINDINGS)
    return()
  endif()
  spectre_add_python_test(${TEST_NAME} ${FILE} "${TAGS}" ${PY_MODULE_DEPENDENCY})
endfunction()

# Add a convenient target name for the pybindings.
add_custom_target(cli)
add_dependencies(cli all-pybindings)
