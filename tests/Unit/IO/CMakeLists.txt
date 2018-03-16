# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_IO")

set(LIBRARY_SOURCES
  Test_H5.cpp
  )

add_test_library(
  ${LIBRARY}
  "IO"
  "${LIBRARY_SOURCES}"
  "IO"
  )
