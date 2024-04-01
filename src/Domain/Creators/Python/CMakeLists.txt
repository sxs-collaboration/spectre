# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDomainCreators")

spectre_python_add_module(
  Creators
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "Domain"
  SOURCES
  Bindings.cpp
  Cylinder.cpp
  DomainCreator.cpp
  Rectilinear.cpp
  Sphere.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Cylinder.hpp
  DomainCreator.hpp
  Rectilinear.hpp
  Sphere.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainCreators
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyCoordinateMaps
  PyDomain
  )
