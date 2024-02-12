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
  GhostDataForReconstruction.hpp
  Inactive.hpp
  InitialTciData.hpp
  Interpolators.hpp
  Jacobians.hpp
  Mesh.hpp
  MethodOrder.hpp
  ObserverCoordinates.hpp
  ObserverMesh.hpp
  ObserverMeshVelocity.hpp
  OnSubcellFaces.hpp
  OnSubcells.hpp
  ReconstructionOrder.hpp
  Reconstructor.hpp
  SubcellOptions.hpp
  SubcellSolver.hpp
  Tags.hpp
  TciCallsSinceRollback.hpp
  TciGridHistory.hpp
  TciStatus.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Mesh.cpp
  MethodOrder.cpp
  ObserverMesh.cpp
  ObserverMeshVelocity.cpp
  TciStatus.cpp
  )
