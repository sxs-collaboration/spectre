# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Ccz4")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Ccz4/"
  "${LIBRARY_SOURCES}"
  "DataBoxTestHelpers;DataStructures"
  )
