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
  Hypercube.cpp
  InitialElementIds.cpp
  Neighbors.cpp
  OrientationMap.cpp
  OrientationMapHelpers.cpp
  SegmentId.cpp
  Side.cpp
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
  Hypercube.hpp
  IndexToSliceAt.hpp
  InitialElementIds.hpp
  MaxNumberOfNeighbors.hpp
  Neighbors.hpp
  OrientationMap.hpp
  OrientationMapHelpers.hpp
  SegmentId.hpp
  Side.hpp
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
