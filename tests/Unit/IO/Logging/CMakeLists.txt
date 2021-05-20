# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Logging")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  Test_Verbosity.cpp
  )

add_test_library(
  ${LIBRARY}
  "IO/Logging/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Logging
  Utilities
  )
