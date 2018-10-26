# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataStructures")

set(LIBRARY_SOURCES
  Test_DataVector.cpp
  Test_DenseMatrix.cpp
  Test_DenseVector.cpp
  Test_FixedHashMap.cpp
  Test_GeneralIndexIterator.cpp
  Test_IdPair.cpp
  Test_Index.cpp
  Test_IndexIterator.cpp
  Test_LeviCivitaIterator.cpp
  Test_OrientVariablesOnSlice.cpp
  Test_SliceIterator.cpp
  Test_SpinWeighted.cpp
  Test_StripeIterator.cpp
  Test_Variables.cpp
  Test_VectorImplHelper.cpp
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
