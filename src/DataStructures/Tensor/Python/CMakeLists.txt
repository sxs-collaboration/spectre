# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyTensor")

spectre_python_add_module(
  Tensor
  MODULE_PATH "DataStructures"
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  # Tensor instantiations are split into multiple files so they can be compiled
  # in parallel and use less memory
  InstantiateScalar.cpp
  InstantiateTensor1d.cpp
  InstantiateTensor2d.cpp
  InstantiateTensor3d.cpp
  PYTHON_FILES
  __init__.py
  Frame.py
  tnsr.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tensor.hpp
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
  )
