# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDgSubcell")

spectre_python_add_module(
  DgSubcell
  MODULE_PATH "Evolution"
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
)

# spectre_python_headers(
#   ${LIBRARY}
#   INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
#   HEADERS
# )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DgSubcell
  pybind11::module
  Spectral
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyCoordinateMaps
  PyDataStructures
  PyTensor
  )
