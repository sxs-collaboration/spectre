# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions")

set(LIBRARY_SOURCES
  Test_TensorExpressions.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions"
  "${LIBRARY_SOURCES}"
  "ErrorHandling"
  )
