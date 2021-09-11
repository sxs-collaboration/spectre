# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT YAPF_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(YAPF_ROOT "")
  set(YAPF_ROOT $ENV{YAPF_ROOT})
endif()

find_package(Python)
get_filename_component(PYTHON_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)

# Look for an executable called yapf
find_program(
  YAPF_EXECUTABLE
  NAMES yapf
  HINTS
  ${YAPF_ROOT}
  ${PYTHON_BIN_DIR}
  DOC "Path to yapf executable")

execute_process(COMMAND "${YAPF_EXECUTABLE}" "--version"
  RESULT_VARIABLE VERSION_RESULT
  OUTPUT_VARIABLE VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

include(FindPackageHandleStandardArgs)

if(VERSION_RESULT MATCHES 0)
  string(REGEX MATCH "[0-9]+\.[0-9]+[\.]?[0-9]*"
    YAPF_VERSION ${VERSION_OUTPUT})
endif(VERSION_RESULT MATCHES 0)

# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Yapf
  REQUIRED_VARS YAPF_EXECUTABLE YAPF_VERSION
  VERSION_VAR YAPF_VERSION)
