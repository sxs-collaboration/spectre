# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PySpectral")

spectre_python_add_module(
  Spectral
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  LogicalCoordinates.cpp
  Mesh.cpp
  Spectral.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  LogicalCoordinates.hpp
  Mesh.hpp
  Spectral.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Spectral
  pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
