# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT BREATHE_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(BREATHE_ROOT "")
  set(BREATHE_ROOT $ENV{BREATHE_ROOT})
endif()

# Look for an executable called breathe-apidoc
find_program(
  BREATHE_APIDOC_EXECUTABLE
  NAMES breathe-apidoc
  PATHS ${BREATHE_ROOT}
  DOC "Path to breathe-apidoc executable")

execute_process(COMMAND "${BREATHE_APIDOC_EXECUTABLE}" "--version"
  RESULT_VARIABLE VERSION_RESULT
  OUTPUT_VARIABLE VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

include(FindPackageHandleStandardArgs)

if(VERSION_RESULT MATCHES 0)
  string(REGEX MATCH "[0-9]+\.[0-9]+[\.]?[0-9]*"
    BREATHE_APIDOC_VERSION ${VERSION_OUTPUT})
  set(BREATHE_VERSION ${BREATHE_APIDOC_VERSION})
endif(VERSION_RESULT MATCHES 0)

# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Breathe
  REQUIRED_VARS BREATHE_APIDOC_EXECUTABLE BREATHE_VERSION
  BREATHE_APIDOC_VERSION
  VERSION_VAR BREATHE_VERSION)
