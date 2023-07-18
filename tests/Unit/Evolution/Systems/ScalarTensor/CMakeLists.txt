# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarTensor")

set(LIBRARY_SOURCES
  Test_Sources.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ScalarTensor"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Framework
  ScalarTensor
  )
