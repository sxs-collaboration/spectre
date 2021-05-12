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
  Events/Test_ObserveFields.cpp
  Test_ActiveGrid.cpp
  Test_CartesianFluxDivergence.cpp
  Test_ComputeBoundaryTerms.cpp
  Test_Matrices.cpp
  Test_Mesh.cpp
  Test_NeighborData.cpp
  Test_NeighborReconstructedFaceSolution.cpp
  Test_PerssonTci.cpp
  Test_PrepareNeighborData.cpp
  Test_Projection.cpp
  Test_RdmpTci.cpp
  Test_Reconstruction.cpp
  Test_SliceData.cpp
  Test_SubcellOptions.cpp
  Test_Tags.cpp
  Test_TciStatus.cpp
  Test_TwoMeshRdmpTci.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DgSubcell/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DgSubcell
  DgSubcellHelpers
  ErrorHandling
  IO
  Parallel
  Spectral
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
