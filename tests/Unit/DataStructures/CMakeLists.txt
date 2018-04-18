# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataStructures")

set(LIBRARY_SOURCES
  Test_DataBox.cpp
  Test_DataBoxTag.cpp
  Test_DataVector.cpp
  Test_Deferred.cpp
  Test_GeneralIndexIterator.cpp
  Test_Identity.cpp
  Test_Index.cpp
  Test_IndexIterator.cpp
  Test_Matrix.cpp
  Test_OrientVariablesOnSlice.cpp
  Test_SliceIterator.cpp
  Test_StripeIterator.cpp
  Test_Tensor.cpp
  Test_TensorExpressions.cpp
  Test_Variables.cpp
  )

add_subdirectory(
  TensorEagerMath
  )

add_test_library(
  ${LIBRARY}
  "DataStructures"
  "${LIBRARY_SOURCES}"
  "DataStructures"
  )
