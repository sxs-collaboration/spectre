# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DgSubcell")

set(LIBRARY_SOURCES
  Actions/Test_Initialize.cpp
  Actions/Test_ReconstructionCommunication.cpp
  Actions/Test_SelectNumericalMethod.cpp
  Actions/Test_TakeTimeStep.cpp
  Actions/Test_TciAndRollback.cpp
  Actions/Test_TciAndSwitchToDg.cpp
  Test_ActiveGrid.cpp
  Test_BackgroundGrVars.cpp
  Test_CartesianFluxDivergence.cpp
  Test_CellCenteredFlux.cpp
  Test_ComputeBoundaryTerms.cpp
  Test_CorrectPackagedData.cpp
  Test_GetActiveTag.cpp
  Test_GetTciDecision.cpp
  Test_GhostData.cpp
  Test_GhostZoneLogicalCoordinates.cpp
  Test_InitialTciData.cpp
  Test_JacobianCompute.cpp
  Test_Matrices.cpp
  Test_Mesh.cpp
  Test_MeshForGhostData.cpp
  Test_NeighborRdmpAndVolumeData.cpp
  Test_NeighborReconstructedFaceSolution.cpp
  Test_NeighborTciDecision.cpp
  Test_PerssonTci.cpp
  Test_PrepareNeighborData.cpp
  Test_Projection.cpp
  Test_RdmpTci.cpp
  Test_RdmpTciData.cpp
  Test_Reconstruction.cpp
  Test_ReconstructionMethod.cpp
  Test_ReconstructionOrder.cpp
  Test_SetInterpolators.cpp
  Test_SliceData.cpp
  Test_SliceTensor.cpp
  Test_SliceVariable.cpp
  Test_SubcellOptions.cpp
  Test_Tags.cpp
  Test_TwoMeshRdmpTci.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  CoordinateMaps
  DataStructures
  DataStructuresHelpers
  DiscontinuousGalerkin
  Domain
  DomainCreators
  DomainStructure
  DomainTimeDependence
  DgSubcell
  DgSubcellHelpers
  ErrorHandling
  Evolution
  FiniteDifference
  GeneralRelativitySolutions
  Hydro
  Parallel
  RelativisticEulerSolutions
  Spectral
  Time
  Utilities
  )
