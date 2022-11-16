# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDomain")

spectre_python_add_module(
  Domain
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  ElementId.cpp
  SegmentId.cpp
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementId.hpp
  SegmentId.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainStructure
  pybind11::module
  Utilities
  )
