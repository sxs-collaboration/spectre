# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActiveGrid.hpp
  Coordinates.hpp
  DidRollback.hpp
  Inactive.hpp
  Jacobians.hpp
  Mesh.hpp
  NeighborData.hpp
  OnSubcellFaces.hpp
  OnSubcells.hpp
  SubcellOptions.hpp
  SubcellSolver.hpp
  Tags.hpp
  TciGridHistory.hpp
  TciStatus.hpp
  )
