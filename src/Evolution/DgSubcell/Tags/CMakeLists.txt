# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActiveGrid.hpp
  CellCenteredFlux.hpp
  Coordinates.hpp
  DataForRdmpTci.hpp
  DidRollback.hpp
  Inactive.hpp
  Jacobians.hpp
  Mesh.hpp
  MethodOrder.hpp
  GhostDataForReconstruction.hpp
  ObserverCoordinates.hpp
  ObserverMesh.hpp
  OnSubcellFaces.hpp
  OnSubcells.hpp
  ReconstructionOrder.hpp
  SubcellOptions.hpp
  SubcellSolver.hpp
  Tags.hpp
  TciGridHistory.hpp
  TciStatus.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ObserverMesh.cpp
  Mesh.cpp
  MethodOrder.cpp
  TciStatus.cpp
  )
