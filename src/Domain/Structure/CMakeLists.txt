# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainStructure)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BlockNeighbor.cpp
  ChildSize.cpp
  CreateInitialMesh.cpp
  Direction.cpp
  Element.cpp
  ElementId.cpp
  ExcisionSphere.cpp
  Hypercube.cpp
  InitialElementIds.cpp
  Neighbors.cpp
  ObjectLabel.cpp
  OrientationMap.cpp
  OrientationMapHelpers.cpp
  SegmentId.cpp
  Side.cpp
  ZCurve.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BlockId.hpp
  BlockNeighbor.hpp
  ChildSize.hpp
  CreateInitialMesh.hpp
  Direction.hpp
  DirectionMap.hpp
  Element.hpp
  ElementId.hpp
  ExcisionSphere.hpp
  Hypercube.hpp
  IndexToSliceAt.hpp
  InitialElementIds.hpp
  MaxNumberOfNeighbors.hpp
  Neighbors.hpp
  ObjectLabel.hpp
  OrientationMap.hpp
  OrientationMapHelpers.hpp
  SegmentId.hpp
  Side.hpp
  TrimMap.hpp
  ZCurve.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  Parallel
  PUBLIC
  DataStructures
  ErrorHandling
  Spectral
  Utilities
  )
