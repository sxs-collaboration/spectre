# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Xcts")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Xcts/"
  "${LIBRARY_SOURCES}"
  ""
  )
