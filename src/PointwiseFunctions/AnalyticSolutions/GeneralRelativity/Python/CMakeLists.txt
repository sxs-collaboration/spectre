# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyGrSolutions")

spectre_python_add_module(
  GeneralRelativity
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticSolutions"
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  GeneralRelativitySolutions
  pybind11::module
  Utilities
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
)
