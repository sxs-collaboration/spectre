# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_ScalarAdvection)

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ScalarAdvection"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ScalarAdvection
  DataStructures
  Utilities
  )
