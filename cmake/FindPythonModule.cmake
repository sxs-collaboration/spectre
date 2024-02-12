# From: https://github.com/ivansafrin/Polycode/

# Find if a Python module is installed
# Found at http://www.cmake.org/pipermail/cmake/2011-January/041666.html
# To use do: find_python_module(PyQt4 REQUIRED) # if required
#        or: find_python_module(PyQt4) # if optional, check PY_PyQt4_FOUND
#            if(PY_PyQt4_FOUND)
#              # do stuff...
#            endif()
function(find_python_module module)
  # Terminate early if the package has already been found
  if(PY_${module}_FOUND)
    return()
  endif()
  cmake_parse_arguments(ARG "REQUIRED" "" "" ${ARGN})
  # Try to import the module and get its location, if it's not already cached
  if(NOT PY_${module}_LOCATION)
    # A module's location is usually a directory, but for binary modules
    # it's a .so file.
    execute_process(COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${PYTHONPATH} ${Python_EXECUTABLE} "-c"
        "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
        RESULT_VARIABLE _${module}_status
        OUTPUT_VARIABLE _${module}_location
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_${module}_status EQUAL 0)
      set(PY_${module}_LOCATION ${_${module}_location} CACHE STRING
          "Location of Python module ${module}")
    endif()
  endif()
  # Make `find_package_handle_standard_args` error if the package is not found
  if(ARG_REQUIRED)
    set(${module}_FIND_REQUIRED TRUE)
  endif()
  find_package_handle_standard_args(PY_${module}
    REQUIRED_VARS PY_${module}_LOCATION)
  set(PY_${module}_FOUND ${PY_${module}_FOUND} PARENT_SCOPE)
endfunction(find_python_module)
