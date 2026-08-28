# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyEvolutionSystemsGeneralizedHarmonic")

spectre_python_add_module(
  GeneralizedHarmonic
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
  GeneralizedHarmonic
  pybind11::module
  PythonBindings
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
