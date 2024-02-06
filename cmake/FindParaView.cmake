# Distributed under the MIT License.
# See LICENSE.txt for details.

find_program(PVPYTHON_EXEC pvpython)

# Get version and runtime environment variables
if (PVPYTHON_EXEC)
  execute_process(
    COMMAND ${PVPYTHON_EXEC} --print --version
    OUTPUT_VARIABLE _OUTPUT
    ERROR_VARIABLE _OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" _OUTPUT "${_OUTPUT}")
  list(LENGTH _OUTPUT _OUTPUT_LENGTH)
  if (_OUTPUT_LENGTH EQUAL 1)
    # '--print' is not supported, just get the version
    list(GET _OUTPUT 0 PARAVIEW_VERSION)
  elseif(_OUTPUT_LENGTH EQUAL 2)
    # '--print' is supported, get the version and environment variables
    list(GET _OUTPUT 0 PARAVIEW_PYTHON_ENV_VARS)
    list(GET _OUTPUT 1 PARAVIEW_VERSION)
  endif()
  string(REPLACE "paraview version " "" PARAVIEW_VERSION "${PARAVIEW_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ParaView
  REQUIRED_VARS PVPYTHON_EXEC
  VERSION_VAR PARAVIEW_VERSION)

# Get Python environment variables
execute_process(
  COMMAND ${PVPYTHON_EXEC} -m paraview.inspect
  OUTPUT_VARIABLE _OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _RESULT
  ERROR_QUIET)
if (_RESULT EQUAL 0)
  string(REPLACE "\n" ";" _OUTPUT "${_OUTPUT}")
  list(GET _OUTPUT 0 PARAVIEW_PYTHON_VERSION)
  list(GET _OUTPUT 1 PARAVIEW_PYTHONPATH)
  string(REPLACE "version: " ""
    PARAVIEW_PYTHON_VERSION "${PARAVIEW_PYTHON_VERSION}")
  string(REPLACE "pythonpath entry: " ""
    PARAVIEW_PYTHONPATH "${PARAVIEW_PYTHONPATH}")
endif()
