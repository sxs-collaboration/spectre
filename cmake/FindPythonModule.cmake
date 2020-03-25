# From: https://github.com/ivansafrin/Polycode/

include(SpectreFindPython)
spectre_find_python(REQUIRED COMPONENTS Interpreter)

# Find if a Python module is installed
# Found at http://www.cmake.org/pipermail/cmake/2011-January/041666.html
# To use do: find_python_module(PyQt4 TRUE) # if required
#        or: find_python_module(PyQt4 FALSE) # if optional, check PY_PYQT4
#            if(PY_PYQT4)
#              # do stuff...
#            endif()
function(find_python_module module is_required)
  string(TOUPPER ${module} module_upper)
  if(NOT PY_${module_upper})
    if(ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
      set(${module}_FIND_REQUIRED TRUE)
    endif()
    # A module's location is usually a directory, but for binary modules
    # it's a .so file.
    get_property(PYTHON_EXEC TARGET Python::Interpreter
      PROPERTY OUTPUT_LOCATION)
    execute_process(COMMAND "${PYTHON_EXEC}" "-c"
        "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
        RESULT_VARIABLE _${module}_status
        OUTPUT_VARIABLE _${module}_location
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _${module}_status)
      set(PY_${module_upper} ${_${module}_location} CACHE STRING
          "Location of Python module ${module}")
    endif(NOT _${module}_status)
  endif(NOT PY_${module_upper})
  find_package_handle_standard_args(PY_${module} DEFAULT_MSG PY_${module_upper})
  if(is_required AND NOT PY_${module_upper})
    message(FATAL_ERROR "Failed to find python module: ${module}")
  endif()
endfunction(find_python_module)
