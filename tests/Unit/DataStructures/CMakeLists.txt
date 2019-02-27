# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataStructures")

set(LIBRARY_SOURCES
  Test_ComplexDataVector.cpp
  Test_ComplexDiagonalModalOperator.cpp
  Test_ComplexModalVector.cpp
  Test_DataVector.cpp
  Test_DenseMatrix.cpp
  Test_DenseVector.cpp
  Test_DiagonalModalOperator.cpp
  Test_FixedHashMap.cpp
  Test_GeneralIndexIterator.cpp
  Test_IdPair.cpp
  Test_Index.cpp
  Test_IndexIterator.cpp
  Test_LeviCivitaIterator.cpp
  Test_ModalVector.cpp
  Test_OrientVariables.cpp
  Test_SliceIterator.cpp
  Test_SpinWeighted.cpp
  Test_StripeIterator.cpp
  Test_Tags.cpp
  Test_TempBuffer.cpp
  Test_Variables.cpp
  Test_VectorImpl.cpp
  Test_VectorImplTestHelper.cpp
  )

add_subdirectory(DataBox)
add_subdirectory(Tensor)

add_test_library(
  ${LIBRARY}
  "DataStructures"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )

spectre_add_python_test(
  "Unit.DataStructures.Python.DataVector"
  Test_DataVector.py
  "unit;datastructures;python")
spectre_add_python_test(
  "Unit.DataStructures.Python.Matrix"
  Test_Matrix.py
  "unit;datastructures;python")
