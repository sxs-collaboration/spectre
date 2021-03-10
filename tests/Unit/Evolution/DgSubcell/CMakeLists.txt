# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DgSubcell")

set(LIBRARY_SOURCES
  Test_ActiveGrid.cpp
  Test_Matrices.cpp
  Test_Mesh.cpp
  Test_Projection.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DgSubcell/"
  "${LIBRARY_SOURCES}"
  "DataStructures;DgSubcell;DgSubcellHelpers"
  )
