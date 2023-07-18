# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ObserverHelpers")

set(LIBRARY_SOURCES
  Test_MockH5.cpp
  Test_MockWriteReductionDataRow.cpp
  )

add_test_library(
  ${LIBRARY}
  "Helpers/Tests/IO/Observers"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ObserverHelpers
  )
