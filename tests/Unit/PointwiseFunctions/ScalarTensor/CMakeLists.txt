# Distributed under the MIT License.
# See LICENSE.txt for details.
set(LIBRARY "Test_ScalarTensorPointwise")

set(LIBRARY_SOURCES
  Test_ScalarCharge.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")
target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Framework
  ScalarTensor
  ScalarTensorPointwise
  )
