# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyCoordinateMaps")

spectre_python_add_module(
  CoordinateMaps
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "Domain"
  SOURCES
  Bindings.cpp
  Composition.cpp
  CoordinateMap.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Composition.hpp
  CoordinateMap.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  CoordinateMaps
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
