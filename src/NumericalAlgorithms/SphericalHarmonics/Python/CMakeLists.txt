# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PySphericalHarmonics")

spectre_python_add_module(
  SphericalHarmonics
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  Strahlkorper.cpp
  StrahlkorperFunctions.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Strahlkorper.hpp
  StrahlkorperFunctions.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  SphericalHarmonics
  SphericalHarmonicsIO
  pybind11::module
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
