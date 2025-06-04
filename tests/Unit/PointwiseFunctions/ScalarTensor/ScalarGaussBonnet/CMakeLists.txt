# Distributed under the MIT License.
# See LICENSE.txt for details.
set(LIBRARY "Test_ScalarGaussBonnetPointwise")

set(LIBRARY_SOURCES
  Test_ScalarSource.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")
target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstraintDamping
  Framework
  ScalarGaussBonnetPointwise
  ScalarTensor
  ScalarTensorPointwise
  )
