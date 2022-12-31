# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDomain")

spectre_python_add_module(
  Domain
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  BlockLogicalCoordinates.cpp
  Domain.cpp
  ElementId.cpp
  ElementLogicalCoordinates.cpp
  FunctionsOfTime.cpp
  InertialCoordinates.cpp
  SegmentId.cpp
  PYTHON_FILES
  __init__.py
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BlockLogicalCoordinates.hpp
  Domain.hpp
  ElementId.hpp
  ElementLogicalCoordinates.hpp
  FunctionsOfTime.hpp
  InertialCoordinates.hpp
  SegmentId.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  FunctionsOfTime
  pybind11::module
  Utilities
  )
