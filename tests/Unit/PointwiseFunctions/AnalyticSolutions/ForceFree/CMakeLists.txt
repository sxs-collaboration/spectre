# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ForceFreeSolutions")

set(LIBRARY_SOURCES
  Test_AlfvenWave.cpp
  Test_ExactWald.cpp
  Test_FastWave.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ForceFreeSolutions
  Options
  Utilities
  )
