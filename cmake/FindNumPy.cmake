# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(PythonInterp REQUIRED)

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
  "import numpy as n; print(n.__version__); print(n.get_include());"
  RESULT_VARIABLE RESULT
  OUTPUT_VARIABLE OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(RESULT MATCHES 0)
  string(REGEX REPLACE ";" "\\\\;" VALUES ${OUTPUT})
  string(REGEX REPLACE "\r?\n" ";" VALUES ${VALUES})
  list(GET VALUES 0 NUMPY_VERSION)
  list(GET VALUES 1 NUMPY_INCLUDE_DIRS)

  string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" __ver_check "${NUMPY_VERSION}")
  if("${__ver_check}" STREQUAL "")
   unset(NUMPY_VERSION)
   unset(NUMPY_INCLUDE_DIRS)
   message(STATUS "Failed to retrieve NumPy version and include path, but got instead:\n${OUTPUT}\n")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy REQUIRED_VARS NUMPY_INCLUDE_DIRS NUMPY_VERSION
                                        VERSION_VAR   NUMPY_VERSION)

if(NUMPY_FOUND)
  message(STATUS "NumPy ver. ${NUMPY_VERSION} found (include: ${NUMPY_INCLUDE_DIRS})")
else()
  message(STATUS "NumPy not found!")
endif()
