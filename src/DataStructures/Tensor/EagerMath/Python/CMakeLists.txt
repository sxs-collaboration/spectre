# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyTensorEagerMath")

spectre_python_add_module(
  EagerMath
  MODULE_PATH "DataStructures/Tensor"
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  Determinant.cpp
  Magnitude.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Determinant.hpp
  Magnitude.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  pybind11::module
  PythonBindings
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
