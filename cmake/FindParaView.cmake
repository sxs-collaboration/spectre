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
  foreach(_ENTRY ${_OUTPUT})
    # Extract the ParaView version from the output
    string(FIND "${_ENTRY}" "paraview version " _FOUND)
    if(NOT ${_FOUND} EQUAL -1)
      set(PARAVIEW_VERSION ${_ENTRY})
    endif()

    # On some machines ParaView needs specific environment variables set, e.g.
    # on CaltechHPC we need to set LD_LIBRARY_PATH. If other env variables
    # need to be set, then we need to possibly update this.
    string(FIND "${_ENTRY}" "LD_LIBRARY_PATH=" _FOUND)
    if(NOT ${_FOUND} EQUAL -1)
      set(PARAVIEW_PYTHON_ENV_VARS ${_ENTRY})
    endif()
  endforeach()
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
else()
  # If for whatever reason pvpython call didn't work, try another search
  get_filename_component(PVPYTHON_BINDIR ${PVPYTHON_EXEC} DIRECTORY)
  get_filename_component(PVPYTHON_BASEDIR ${PVPYTHON_BINDIR} DIRECTORY)
  file(GLOB_RECURSE
    PARAVIEW_PYTHONPATH
    ${PVPYTHON_BASEDIR}
    "${PVPYTHON_BASEDIR}/lib*/python*/site-packages/paraview/simple.py"
  )
  find_file(PARAVIEW_PYTHONPATH
    NAMES "simple.py"
    PATHS "${PVPYTHON_BASEDIR}"
    PATH_SUFFIXES "/.*"
    REQUIRED)
  # go back up to the site-packages
  get_filename_component(PARAVIEW_PYTHONPATH ${PARAVIEW_PYTHONPATH} DIRECTORY)
  get_filename_component(PARAVIEW_PYTHONPATH ${PARAVIEW_PYTHONPATH} DIRECTORY)
endif()
