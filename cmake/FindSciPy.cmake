# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(PythonInterp REQUIRED)

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
  "import scipy as s; print(s.__version__); print(s.get_include());"
  RESULT_VARIABLE RESULT
  OUTPUT_VARIABLE OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(RESULT MATCHES 0)
  string(REGEX REPLACE ";" "\\\\;" VALUES ${OUTPUT})
  string(REGEX REPLACE "\r?\n" ";" VALUES ${VALUES})
  list(GET VALUES 0 SCIPY_VERSION)
  list(GET VALUES 1 SCIPY_INCLUDE_DIRS)

  string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" __ver_check "${SCIPY_VERSION}")
  if("${__ver_check}" STREQUAL "")
   unset(SCIPY_VERSION)
   unset(SCIPY_INCLUDE_DIRS)
   message(STATUS "Failed to retrieve SciPy version and include path, but got instead:\n${OUTPUT}\n")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SciPy REQUIRED_VARS SCIPY_INCLUDE_DIRS SCIPY_VERSION
                                        VERSION_VAR   SCIPY_VERSION)

if(SCIPY_FOUND)
  message(STATUS "SciPy ver. ${SCIPY_VERSION} found (include: ${SCIPY_INCLUDE_DIRS})")
else()
  message(STATUS "SciPy not found!")
endif()
