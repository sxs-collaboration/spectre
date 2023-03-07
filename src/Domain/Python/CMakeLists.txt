# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyDomain")

spectre_python_add_module(
  Domain
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  Block.cpp
  BlockLogicalCoordinates.cpp
  Domain.cpp
  ElementId.cpp
  ElementLogicalCoordinates.cpp
  ElementMap.cpp
  FunctionsOfTime.cpp
  JacobianDiagnostic.cpp
  RadiallyCompressedCoordinates.cpp
  SegmentId.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Block.hpp
  BlockLogicalCoordinates.hpp
  Domain.hpp
  ElementId.hpp
  ElementLogicalCoordinates.hpp
  ElementMap.hpp
  FunctionsOfTime.hpp
  JacobianDiagnostic.hpp
  RadiallyCompressedCoordinates.hpp
  SegmentId.hpp
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  FunctionsOfTime
  pybind11::module
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyCoordinateMaps
  PyDataStructures
  PyTensor
  )
