# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyPuncturesPointwiseFunctions")

spectre_python_add_module(
  Punctures
  MODULE_PATH "PointwiseFunctions"
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  PuncturesPointwiseFunctions
  pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
