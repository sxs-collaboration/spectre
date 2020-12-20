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
  Tensor/TensorData.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  pybind11::module
  PyBindings
  Utilities
  )
