# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DgSubcell")

set(LIBRARY_SOURCES
  Actions/Test_Initialize.cpp
  Actions/Test_SelectNumericalMethod.cpp
  Actions/Test_TciAndRollback.cpp
  Test_ActiveGrid.cpp
  Test_Matrices.cpp
  Test_Mesh.cpp
  Test_NeighborData.cpp
  Test_PerssonTci.cpp
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
  Parallel
  Spectral
  )
