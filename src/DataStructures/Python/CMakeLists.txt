# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDataStructures")

spectre_python_add_module(
  DataStructures
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  DataVector.cpp
  Matrix.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DataVector.hpp
  Matrix.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  pybind11::module
  PythonBindings
  Utilities
  )
