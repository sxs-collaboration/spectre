# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run Python tests we need the development component (libs and include dirs).
# It provides the `Python::Python` imported target. We find the interpreter
# component as well to make sure the find is consistent with earlier finds that
# only looked for the interpreter, possibly guided by the Python_EXECUTABLE
# variable set by the user.
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# We can't rely on CMake 3.14 quite yet so we backport the `Python::NumPy`
# imported target that FindPython in CMake 3.14+ could also provide.
find_package(NumPy 1.10 REQUIRED)

message(STATUS "NumPy incl: " ${NUMPY_INCLUDE_DIRS})
message(STATUS "NumPy vers: " ${NUMPY_VERSION})

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "NumPy Version: ${NUMPY_VERSION}\n"
  )

# Also check that SciPy is installed
find_python_module(scipy TRUE)

add_library(Python::NumPy INTERFACE IMPORTED)
set_property(TARGET Python::NumPy PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${NUMPY_INCLUDE_DIRS})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Python::NumPy Python::Python
  )
