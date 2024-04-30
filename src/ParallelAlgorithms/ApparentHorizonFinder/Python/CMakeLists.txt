# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyApparentHorizonFinder")

spectre_python_add_module(
  ApparentHorizonFinder
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
  ApparentHorizonFinder
  SphericalHarmonics
  pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PySphericalHarmonics
)
