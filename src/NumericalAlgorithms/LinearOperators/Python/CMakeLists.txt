# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyLinearOperators")

spectre_python_add_module(
  LinearOperators
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "NumericalAlgorithms"
  SOURCES
  Bindings.cpp
  DefiniteIntegral.cpp
  PartialDerivatives.cpp
  PowerMonitors.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DefiniteIntegral.hpp
  PartialDerivatives.hpp
  PowerMonitors.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  LinearOperators
  pybind11::module
  PythonBindings
  Spectral
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PySpectral
  PyTensor
  )
