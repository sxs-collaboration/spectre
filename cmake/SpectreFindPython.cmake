# Distributed under the MIT License.
# See LICENSE.txt for details.

function(spectre_find_python)
  # Finds Python components and adds targets for them.
  #
  # Available components:
  # - Development: the Python libraries target: Python::Development
  # - Interpreter: the Python interpreter target: Python::Interpreter
  # - NumPy: NumPy library target: Python::NumPy
  # - SciPy: SciPy library target: Python::SciPy
  #
  # If a component must be found, you can pass `REQUIRED`.
  #
  # We provide this function as a wrapper around all the Python components
  # that we may need, and also to support both newer and older versions of
  # CMake since they find Python differently.
  set(KNOWN_COMPONENTS
    Development
    Interpreter
    NumPy
    SciPy)

  # Set up supported arguments. REQUIRED is optional, so if not specified
  # it is FALSE. COMPONENTS must be one of KNOWN_COMPONENTS.
  set(MULTI_VALUE_ARGS COMPONENTS)
  cmake_parse_arguments(
    ARG "REQUIRED"
    ""
    "${MULTI_VALUE_ARGS}"
    ${ARGN})

  # Make sure we have received reasonable arguments.
  list(LENGTH ARG_COMPONENTS ARG_COMPONENTS_LENGTH)
  if(${ARG_COMPONENTS_LENGTH} EQUAL 0)
    message(FATAL_ERROR
      "Must pass at least one component to spectre_find_python. Available "
      "components are: ${KNOWN_COMPONENTS}.")
  endif(${ARG_COMPONENTS_LENGTH} EQUAL 0)

  foreach(REQUESTED_COMPONENT ${ARG_COMPONENTS})
    list(FIND KNOWN_COMPONENTS ${REQUESTED_COMPONENT} FOUND_REQUESTED_COMPONENT)
    if(${FOUND_REQUESTED_COMPONENT} EQUAL -1)
      message(FATAL_ERROR
        "Requested Python component ${REQUESTED_COMPONENT} not supported "
        "by spectre_find_python. Please add the functionality.")
    endif(${FOUND_REQUESTED_COMPONENT} EQUAL -1)
  endforeach(REQUESTED_COMPONENT ${ARG_COMPONENTS})

  # We need to find the python interpreter if we are to find NumPy or SciPy.
  # For Python libs it is good to have found the python interpreter so that
  # the libs and the interpreter match.
  if(NOT "Interpreter" IN_LIST ARG_COMPONENTS)
    list(APPEND ARG_COMPONENTS "Interpreter")
  endif(NOT "Interpreter" IN_LIST ARG_COMPONENTS)

  # Find the interpreter and/or the development libraries. Handle older
  # and newer CMake versions separately. We expose targets that are
  # compatible with CMake's FindPython targets.
  if(CMAKE_VERSION VERSION_LESS 3.12)
    if("Interpreter" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Interpreter)
      if(ARG_REQUIRED)
        find_package(PythonInterp REQUIRED)
      else(ARG_REQUIRED)
        find_package(PythonInterp)
      endif(ARG_REQUIRED)

      if(PYTHONINTERP_FOUND)
        add_executable (Python::Interpreter IMPORTED)
        set_property (TARGET Python::Interpreter
          PROPERTY IMPORTED_LOCATION "${PYTHON_EXECUTABLE}")
        get_directory_property(HAS_PARENT PARENT_DIRECTORY)
        set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
        set(Python_Interpreter_FOUND ${PYTHONINTERP_FOUND})
      endif(PYTHONINTERP_FOUND)
    endif("Interpreter" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Interpreter)

    if("Development" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Python)
      if(ARG_REQUIRED)
        find_package(PythonLibs REQUIRED)
      else(ARG_REQUIRED)
        find_package(PythonLibs)
      endif(ARG_REQUIRED)

      if(PYTHONLIBS_FOUND)
        add_library(Python::Python INTERFACE IMPORTED)
        set_property(TARGET Python::Python
          APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${PYTHON_LIBRARIES})
        set_property(TARGET Python::Python PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIRS})
        set(Python_LIBRARIES ${Python_LIBRARIES})
        set(Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
      endif(PYTHONLIBS_FOUND)
    endif("Development" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Python)
  else(CMAKE_VERSION VERSION_LESS 3.12)
    if("Interpreter" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Interpreter)
      if(ARG_REQUIRED)
        find_package(Python COMPONENTS Interpreter REQUIRED)
      else(ARG_REQUIRED)
        find_package(Python COMPONENTS Interpreter)
      endif(ARG_REQUIRED)
      if(Python_Interpreter_FOUND)
        set(PYTHONINTERP_FOUND ${Python_Interpreter_FOUND})
        set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
      endif(Python_Interpreter_FOUND)
    endif("Interpreter" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Interpreter)

    if("Development" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Python)
      if(ARG_REQUIRED)
        find_package(Python COMPONENTS Development REQUIRED)
      else(ARG_REQUIRED)
        find_package(Python COMPONENTS Development)
      endif(ARG_REQUIRED)
      if(Python_Development_FOUND)
        set(PYTHON_LIBRARIES ${Python_LIBRARIES})
        set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
      endif(Python_Development_FOUND)
    endif("Development" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::Python)
  endif(CMAKE_VERSION VERSION_LESS 3.12)

  if("NumPy" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::NumPy)
    # We use our own NumPy find so we can specify a required minimum version.
    if(ARG_REQUIRED)
      find_package(NumPy 1.10 REQUIRED)
    else(ARG_REQUIRED)
      find_package(NumPy 1.10)
    endif(ARG_REQUIRED)

    if(NUMPY_FOUND)
      add_library(Python::NumPy INTERFACE IMPORTED)
      set_property(TARGET Python::NumPy APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${NUMPY_INCLUDE_DIRS})

      message(STATUS "NumPy incl: " ${NUMPY_INCLUDE_DIRS})
      message(STATUS "NumPy vers: " ${NUMPY_VERSION})

      file(APPEND
        "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
        "NumPy Version:  ${NUMPY_VERSION}\n"
        )
    endif(NUMPY_FOUND)
  endif("NumPy" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::NumPy)

  if("SciPy" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::SciPy)
    if(ARG_REQUIRED)
      find_package(SciPy REQUIRED)
    else(ARG_REQUIRED)
      find_package(SciPy)
    endif(ARG_REQUIRED)

    if(SCIPY_FOUND)
      add_library(Python::SciPy INTERFACE IMPORTED)
      set_property(TARGET Python::SciPy APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${SCIPY_INCLUDE_DIRS})

      message(STATUS "SciPy incl: " ${SCIPY_INCLUDE_DIRS})
      message(STATUS "SciPy vers: " ${SCIPY_VERSION})

      file(APPEND
        "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
        "SciPy Version:  ${SCIPY_VERSION}\n"
        )
    endif(SCIPY_FOUND)
  endif("SciPy" IN_LIST ARG_COMPONENTS AND NOT TARGET Python::SciPy)
endfunction(spectre_find_python)
