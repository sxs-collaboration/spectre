# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyEvolutionSystemsScalarWave")

spectre_python_add_module(
  ScalarWave
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "Evolution/Systems"
  SOURCES
  Bindings.cpp
  FilledSpherePowerMonitor.cpp
  SphericalShellPowerMonitor.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FilledSpherePowerMonitor.hpp
  SphericalShellPowerMonitor.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  pybind11::module
  PythonBindings
  ScalarWave
  Spectral
  SphericalHarmonics
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyDomain
  PySpectral
  PyTensor
  )
