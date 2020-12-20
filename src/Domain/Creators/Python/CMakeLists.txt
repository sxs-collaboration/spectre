# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDomainCreators")

spectre_python_add_module(
  Creators
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "Domain"
  SOURCES
  Bindings.cpp
  Brick.cpp
  Cylinder.cpp
  DomainCreator.cpp
  Interval.cpp
  Rectangle.cpp
  Shell.cpp
  Sphere.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainCreators
  pybind11::module
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDomain
  )
