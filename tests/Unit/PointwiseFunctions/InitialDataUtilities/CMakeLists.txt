# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_InitialDataUtilities)

set(LIBRARY_SOURCES
  Tags/Test_InitialData.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/InitialDataUtilities"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Parallel
  )
