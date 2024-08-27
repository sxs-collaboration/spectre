# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyHydro")

spectre_python_add_module(
  Hydro
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
  MODULE_PATH "PointwiseFunctions"
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Hydro
  pybind11::module
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
