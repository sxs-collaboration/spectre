# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActiveGrid.hpp
  BackgroundGrVars.hpp
  CartesianFluxDivergence.hpp
  CellCenteredFlux.hpp
  CombineVolumeGhostData.hpp
  ComputeBoundaryTerms.hpp
  CorrectPackagedData.hpp
  DgSubcell.hpp
  DisableLts.hpp
  GetActiveTag.hpp
  GetTciDecision.hpp
  GhostData.hpp
  GhostZoneInverseJacobian.hpp
  GhostZoneLogicalCoordinates.hpp
  InitialTciData.hpp
  Matrices.hpp
  Mesh.hpp
  NeighborRdmpAndVolumeData.hpp
  NeighborReconstructedFaceSolution.hpp
  NeighborReconstructedFaceSolution.tpp
  NeighborTciDecision.hpp
  PerssonTci.hpp
  PrepareNeighborData.hpp
  Projection.hpp
  RdmpTci.hpp
  RdmpTciData.hpp
  ReceiveSubcellDataForDg.hpp
  Reconstruction.hpp
  ReconstructionMethod.hpp
  ReconstructionOrder.hpp
  SetInterpolators.hpp
  SliceData.hpp
  SliceTensor.hpp
  SliceVariable.hpp
  SubcellEqualRateRegion.hpp
  SubcellOptions.hpp
  TwoMeshRdmpTci.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ActiveGrid.cpp
  CartesianFluxDivergence.cpp
  CombineVolumeGhostData.cpp
  DisableLts.cpp
  GhostData.cpp
  GhostZoneLogicalCoordinates.cpp
  InitialTciData.cpp
  Matrices.cpp
  Mesh.cpp
  NeighborRdmpAndVolumeData.cpp
  NeighborTciDecision.cpp
  PerssonTci.cpp
  Projection.cpp
  RdmpTci.cpp
  RdmpTciData.cpp
  ReceiveSubcellDataForDg.cpp
  Reconstruction.cpp
  ReconstructionMethod.cpp
  SliceData.cpp
  SubcellEqualRateRegion.cpp
  SubcellOptions.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  DomainCreators
  Evolution
  Interpolation
  Parallel
  Spectral

  INTERFACE
  Boost::boost
  Events
  FunctionsOfTime
  Time

  PRIVATE
  BLAS::BLAS
  ErrorHandling
  Serialization
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(Tags)

add_subdirectory(Python)
