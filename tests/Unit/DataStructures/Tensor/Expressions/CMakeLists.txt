# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions")

set(LIBRARY_SOURCES
  Test_AddSubtract.cpp
  Test_Contract.cpp
  Test_Product.cpp
  Test_TensorExpressions.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
