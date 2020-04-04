# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT SPHINX_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(SPHINX_ROOT "")
  set(SPHINX_ROOT $ENV{SPHINX_ROOT})
endif()

# Look for an executable called sphinx-build or sphinx-build2
find_program(
  SPHINX_EXECUTABLE
  NAMES sphinx-build sphinx-build2
  PATHS ${SPHINX_ROOT}
  DOC "Path to sphinx-build or sphinx-build2 executable")

execute_process(COMMAND "${SPHINX_EXECUTABLE}" "--version"
  RESULT_VARIABLE VERSION_RESULT
  OUTPUT_VARIABLE VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(VERSION_RESULT MATCHES 0)
  string(REGEX REPLACE "sphinx-build[2]? " "" SPHINX_VERSION ${VERSION_OUTPUT})
endif(VERSION_RESULT MATCHES 0)

include(FindPackageHandleStandardArgs)
# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Sphinx
  REQUIRED_VARS SPHINX_EXECUTABLE SPHINX_VERSION
  VERSION_VAR SPHINX_VERSION
  )
