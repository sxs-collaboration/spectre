# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PySpectral")

spectre_python_add_module(
  Spectral
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  Mesh.cpp
  Spectral.cpp
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  Spectral
  pybind11::module
)
