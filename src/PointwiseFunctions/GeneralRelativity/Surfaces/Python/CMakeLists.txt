# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyGrSurfaces")

spectre_python_add_module(
  Surfaces
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/GeneralRelativity"
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  GrSurfaces
  pybind11::module
  SphericalHarmonics
  Utilities
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PySphericalHarmonics
  PyTensor
)
