# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FixConservativesAndComputePrims.cpp
  NeighborPackagedData.cpp
  PrimitiveGhostData.cpp
  PrimsAfterRollback.cpp
  ResizeAndComputePrimitives.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FixConservativesAndComputePrims.hpp
  NeighborPackagedData.hpp
  PrimitiveGhostData.hpp
  PrimsAfterRollback.hpp
  ResizeAndComputePrimitives.hpp
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TimeDerivative.hpp
  )
