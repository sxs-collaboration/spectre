# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PySurfaceFinder")

spectre_python_add_module(
  SurfaceFinder
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
  FindRadialSurface.py
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  SphericalHarmonics
  SurfaceFinder
  pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PySphericalHarmonics
)
