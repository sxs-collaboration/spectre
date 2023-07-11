# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SystemUtilities")

set(LIBRARY_SOURCES
  Test_Prefetch.cpp
)

add_test_library(
  ${LIBRARY}
  "Utilities/System"
  "${LIBRARY_SOURCES}"
  ""
)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  SystemUtilities
)
